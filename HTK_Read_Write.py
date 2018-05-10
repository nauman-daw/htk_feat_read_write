# Copyright (c) 2007 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""Read and write HTK feature files.
The original python2 version of the code is taken from: http://my.fit.edu/~vkepuska/ece5526/SPHINX/SphinxTrain/python/sphinx/htkmfc.py
Last Edited by Nauman Dawalatabad (IIT Madras) on 30 April, 2018.
> Code is now compatible with python2 and python3 both.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu> \nNauman Dawalatabad <nauman@cse.iitm.ac.in>"
__version__ = "$Revision $"

from struct import unpack, pack
import numpy
import sys

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11


## For Python 3 
_E = 0o100 # has energy
_N = 0o200 # absolute energy supressed
_D = 0o400 # has delta coefficients
_A = 0o1000 # has acceleration (delta-delta) coefficients
_C = 0o2000 # is compressed
_Z = 0o4000 # has zero mean static coefficients
_K = 0o10000 # has CRC checksum
_O = 0o20000 # has 0th cepstral coefficient
_V = 0o40000 # has VQ data
_T = 0o100000 # has third differential coefficients


def open_file(f, mode=None, veclen=13):
    """Open an HTK format feature file for reading or writing.
    The mode parameter is 'rb' (reading) or 'wb' (writing)."""
    print ("mode---------->", mode)
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return HTKFeat_read(f) # veclen is ignored since it's in the file
    elif mode in ('w', 'wb'):
        return HTKFeat_write(f, veclen)
    else:
        raise Exception("mode must be 'r', 'rb', 'w', or 'wb'") 

class HTKFeat_read(object):
    "Read HTK format feature files"
    def __init__(self, filename=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open_file(filename)

    def __iter__(self):
        self.fh.seek(12,0)
        return self

    def open_file(self, filename):
        self.filename = filename
        self.fh = open(filename, "rb")  
        self.readheader()

    def readheader(self):
        self.fh.seek(0,0)
        spam = self.fh.read(12)
        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = \
                       unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize / 2
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = numpy.fromfile(self.fh, 'f', self.veclen)
                self.B = numpy.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'    
            self.veclen = self.sampSize / 4
        self.hdrlen = self.fh.tell()

    def seek(self, idx):
        self.fh.seek(self.hdrlen + idx * self.sampSize, 0)

    def next(self):
        vec = numpy.fromfile(self.fh, self.dtype, self.veclen)
        if len(vec) == 0:
            raise StopIteration
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self):
        return self.next()

    def getall(self):
        self.seek(0)
        data = numpy.fromfile(self.fh, self.dtype)
        data = data.reshape(int(len(data)/self.veclen) , int(self.veclen))  
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

class HTKFeat_write(object):
    "Write Sphinx-II format feature files"
    def __init__(self, filename=None,
                 veclen=13, sampPeriod=100000,
                 paramKind = (MFCC | _O)):
        self.veclen = veclen
        self.sampPeriod = sampPeriod
        self.sampSize = veclen * 4
        self.paramKind = paramKind
        self.dtype = 'f'
        self.filesize = 0
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open_file(filename)

    def __del__(self):
        self.close()

    def open_file(self, filename):
        self.filename = filename
        self.fh = open(filename, "wb") 
        self.writeheader()

    def close(self):
        self.writeheader()

    def writeheader(self):
        self.fh.seek(0,0)
        self.fh.write(pack(">IIHH", self.filesize,
                           self.sampPeriod,
                           self.sampSize,
                           self.paramKind))

    def writevec(self, vec):
        if len(vec) != self.veclen:
            raise Exception("Vector length must be %d" % self.veclen)
        if self.swap:
            numpy.array(vec, self.dtype).byteswap().tofile(self.fh)
        else:
            numpy.array(vec, self.dtype).tofile(self.fh)
        self.filesize = self.filesize + self.veclen

    def writeall(self, arr):
        for row in arr:
            self.writevec(row)


def my_Main(argv):
    sampleName = "CMU_20020319-1400"  #argv[0]
    featDir = "/SpeakerID-RIC/nauman/meetings_data/features/mfcc/"
    featFile = featDir + sampleName + '.fea'
    obj_read = HTKFeat_read(featFile)
    feats = obj_read.getall()
#    print (feats)

'''
	# For writing feature into htk format
    my_out_file="output_file.fea"
    veclen = 19
    sample_period = 100000
    obj_write = HTKFeat_write(my_out_file, veclen, sample_period)
    obj_write.writeall(feats)
'''

if __name__ == "__main__":
	my_Main(sys.argv)	

