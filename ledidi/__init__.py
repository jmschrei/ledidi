# __init__.py
# Authors: Yang Lu <ylu465@uw.edu> and Jacob Schreiber <jmschreiber91@gmail.com>

from .ledidi import Ledidi
from .ledidi import TensorFlowRegressor

__version__ = '0.2.0'

import numpy
import pyBigWig

from tqdm import tqdm

def sequence_to_ohe(sequence, ignore='N', alphabet=None, dtype='int8', 
	verbose=False, **kwargs):
	"""Converts a string or list of characters into a one-hot encoding.

	This function will take in either a string or a list and convert it into a
	one-hot encoding. If the input is a string, each character is assumed to be
	a different symbol, e.g. 'ACGT' is assumed to be a sequence of four 
	characters. If the input is a list, the elements can be any size.

	Although this function will be used here primarily to convert nucleotide
	sequences into one-hot encoding with an alphabet of size 4, in principle
	this function can be used for any types of sequences.

	Parameters
	----------
	sequence : str or list
		The sequence to convert to a one-hot encoding.

	ignore : str, optional
		A character to indicate setting nothing to 1 for that row, keeping the
		encoding entirely 0's for that row. In the context of genomics, this is
		the N character. Default is 'N'.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet. If None is passed in, the alphabet will be
		determined from the sequence, but this may be time consuming for
		large sequences. Default is None.

	dtype : str or numpy.dtype, optional
		The data type of the returned encoding. Default is int8.

	verbose : bool or str, optional
		Whether to display a progress bar. If a string is passed in, use as the
		name of the progressbar. Default is False.

	kwargs : arguments
		Arguments to be passed into tqdm. Default is None.

	Returns
	-------
	one : numpy.ndarray
		A binary matrix of shape (alphabet_size, sequence_length) where
		alphabet_size is the number of unique elements in the sequence and
		sequence_length is the length of the input sequence.
	"""

	name = None if verbose in (True, False) else verbose
	d = verbose is False

	if isinstance(sequence, str):
		sequence = list(sequence)

	alphabet = alphabet or numpy.unique(sequence)
	alphabet = [char for char in alphabet if char != ignore]
	alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

	ohe = numpy.zeros((len(sequence), len(alphabet)), dtype=dtype)
	for i, char in tqdm(enumerate(sequence), disable=d, desc=name, **kwargs):
		if char != ignore:
			idx = alphabet_lookup[char]
			ohe[i, idx] = 1

	return ohe

def fasta_to_ohe(filename, include_chroms=None, exclude_chroms=None, 
	ignore='N', alphabet=['A', 'C', 'G', 'T', 'N'], dtype='int8', verbose=True):
	"""Read in a FASTA file and output a dictionary of binary encodings.

	This function will take in the path to a FASTA-formatted file and convert
	it to a set of one-hot encodings---one for each chromosome. Optionally,
	the user can specify a set of chromosomes to include or exclude from
	the returned dictionary.

	Parameters
	----------
	filename : str
		The path to the FASTA-formatted file to open.

	include_chroms : set or tuple or list, optional
		The exact names of chromosomes in the FASTA file to include, excluding
		all others. If None, include all chromosomes (except those specified by
		exclude_chroms). Default is None.

	exclude_chroms : set or tuple or list, optional
		The exact names of chromosomes in the FASTA file to exclude, including
		all others. If None, include all chromosomes (or the set specified by
		include_chroms). Default is None.

	ignore : str, optional
		A character to indicate setting nothing to 1 for that row, keeping the
		encoding entirely 0's for that row. In the context of genomics, this is
		the N character. Default is 'N'.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet. If None is passed in, the alphabet will be
		determined from the sequence, but this may be time consuming for
		large sequences. Must include the ignore character. Default is
		['A', 'C', 'G', 'T', 'N'].

	dtype : str or numpy.dtype, optional
		The data type of the returned encoding. Default is int8. 

	verbose : bool or str, optional
		Whether to display a progress bar. If a string is passed in, use as the
		name of the progressbar. Default is False.

	Returns
	-------
	chroms : dict
		A dictionary of one-hot encodings where the keys are the names of the
		chromosomes (exact strings from the header lines in the FASTA file)
		and the values are the one-hot encodings as numpy arrays.
	"""

	sequences = {}
	name, sequence = None, None
	skip_chrom = False

	with open(filename, "r") as infile:
		for line in infile:
			if line.startswith(">"):
				if name is not None and skip_chrom is False:
					sequences[name] = sequence

				sequence = []
				name = line[1:].strip("\n")
				if include_chroms is not None and name not in include_chroms:
					skip_chrom = True
				elif exclude_chroms is not None and name in exclude_chroms:
					skip_chrom = True
				else:
					skip_chrom = False

			else:
				if skip_chrom == False:
					sequence.extend(list(line.rstrip("\n").upper()))

	encodings = {}
	for i, (name, sequence) in enumerate(sequences.items()):
		encodings[name] = sequence_to_ohe(sequence, ignore=ignore, 
			alphabet=alphabet, dtype=dtype, position=i,
			verbose=name if verbose else verbose)

	return encodings

def bigwig_to_arrays(filename, include_chroms=None, exclude_chroms=None, 
	 fillna=0, dtype='float32'):
	"""Read in a bigWig file and output a dictionary of signal arrays.

	This function will take in a filename, open it, and output the
	basepair-resolution signal for the track for all desired chromosomes.

	Parameters
	----------
	filename : str
		The path to the bigWig to open.

	include_chroms : set or tuple or list, optional
		The exact names of chromosomes in the bigWig file to include, excluding
		all others. If None, include all chromosomes (except those specified by
		exclude_chroms). Default is None.

	exclude_chroms : set or tuple or list, optional
		The exact names of chromosomes in the bigWig file to exclude, including
		all others. If None, include all chromosomes (or the set specified by
		include_chroms). Default is None.

	fillna : float or None, optional
		The value to fill NaN values with. If None, keep them as is. Default is 0.

	dtype : str or numpy.dtype, optional
		The data type of the returned encoding. Default is int8.

	Returns
	-------
	signals : dict
		A dictionary of signal values where the keys are the names of the
		chromosomes and the values are arrays of signal values.
	"""

	signals = {}
	chroms = []

	bw = pyBigWig.open(filename, "r")
	for chrom in bw.chroms().keys():
		if include_chroms and chrom not in include_chroms:
			continue
		elif exclude_chroms and chrom in exclude_chroms:
			continue
		else:
			signal = bw.values(chrom, 0, -1, numpy=True).astype(dtype)
			if fillna is not None:
				signal = numpy.nan_to_num(signal, nan=fillna, copy=False)

			signals[chrom] = signal

	return signals