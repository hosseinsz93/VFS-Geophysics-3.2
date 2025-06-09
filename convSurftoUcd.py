#!/home/wayne/bin/pyenv/bin/python3

# perry
###!/home/wayne/bin/pyenv/bin/python3

# seawulf
# module load python/3.8.6
###!/mnt/lustre/wayne/bin/python3/bin/python3

# zagros
###!/mnt/lustre/wayne/bin/python3/bin/python3


# Convert surface files output from vfs into ucd format
# so they can be used to vfs input.

import argparse
import os
import re


DEBUG = False
# DEBUG = True


class ReadFile:
    def __init__(self, fn):
        self.f = open(fn, 'r')
        self.recList = [ ]
        
    def __del__(self):
        self.f.close()

    def readVariable(self):
        self.rec = self.f.readline()
        self.rec = self.rec.strip()
        if self.rec[0:10] != "Variables=":
            exit('First records is not "Variables" type.')

    def readZone(self):
        self.rec = self.f.readline()
        self.rec = self.rec.strip()
        if self.rec[0:4] != 'ZONE':
            exit('Second records is not "ZONE" type.')
        zonere = re.compile("ZONE T='TRIANGLES', N=([0-9]+), E=([0-9]+),")
        ma = zonere.match(self.rec)
        maxVerts = int(ma.group(1))
        maxShells = int(ma.group(2))
        return maxVerts, maxShells

    def readCntFields(self, cnt, returnValue = True):
        if returnValue:
            rList = [ ]
        for i in range(cnt):
            while len(self.recList) == 0:
                self.rec = self.f.readline()
                self.rec = self.rec.strip()
                self.recList = self.rec.split()
            if returnValue:
                rList.append(self.recList[0])
            self.recList.pop(0)
        if returnValue:
            return rList

    
def calcSurfUcd():
    if DEBUG:
        import pdb
        pdb.set_trace()

    args = myparser()

    # make sure there is a / at the end of the destination directory
    # if is was specified.
    if len(args.dir) != 0 and args.dir[len(args.dir)-1:] != os.path.sep:
        args.dir += os.path.sep

    for fn in args.filename:
        print(f'Processing {fn}')

        # Open input file
        f = ReadFile(fn)
        
        # Open output file, remove any prepended direcotories and last .[~]*
        # leaving the filename without a file type.
        idx = fn.rfind(os.path.sep)
        if idx < 0:
            fno = fn
        else:
            fno = fn[idx+1:]
        idx = fno.rfind('.')
        if idx >= 0:
            fno = fno[:idx]
        fno = args.dir + fno + '.ucd'
        
        with open(fno, 'w') as fo:
            fo.write('# Convert vfs output surface file into a ucd\n')
            fo.write('# file that can be input to vfs.\n')
            
            f.readVariable()
            
            maxVerts, maxShells = f.readZone()
            fo.write(f'{maxVerts} {maxShells} 0 0 0\n')

            # Read and output coordinates
            x = f.readCntFields(maxVerts)
            y = f.readCntFields(maxVerts)
            z = f.readCntFields(maxVerts)
            for idx in range(len(x)):
                fo.write(f'{idx+1} {x[idx]} {y[idx]} {z[idx]}\n')

            # Ignore cell based data.
            f.readCntFields(9*maxShells, returnValue = False)

            # Read and output shells
            for i in range(maxShells):
                tmp = f.readCntFields(3)
                fo.write(f'{i+1} 0 tri {tmp[0]} {tmp[1]} {tmp[2]}\n')
                    

                
def myparser():
    mainparser = argparse.ArgumentParser()

    mainparser.add_argument('filename', nargs='+',
                            help='List of file names to be concated')

    mainparser.add_argument('-d', '--dir', help='Output file directory.', default='')

    arg = mainparser.parse_args()
    return (arg)


if __name__ == '__main__':
    calcSurfUcd()
