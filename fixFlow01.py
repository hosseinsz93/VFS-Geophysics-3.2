###!/mnt/lustre/wayne/bin/python3/bin/python3

# perry
###!/home/wayne/pythonenv/bin/python3

# zagros
###!/mnt/lustre/wayne/bin/python3/bin/python3

# seawulf
# module load python/3.8.6
###!pyhton3

import argparse
import re

DEBUG = False
# DEBUG = True


def partTraj():
    if DEBUG:
        import pdb
        pdb.set_trace()

    args = myparser()

    for finame in args.filename:
        print('Reading file {}'.format(finame))

        # Make new file name and open file.
        i1 = finame.rfind('.dat')
        foname = finame[0:i1] + '.fixed' + finame[i1:]

        firstHeader = False
        with open(finame, 'r') as fi:
            with open(foname, 'w') as fo:
                for buff in fi:
                    # is this a blank line?
                    if re.search('^\s*\n$', buff):
                        continue

                    # Keep only the first header
                    if buff[0] == '*':
                        if firstHeader:
                            continue
                        firstHeader = True

                    fo.write(buff)

    
def myparser():
    mainparser = argparse.ArgumentParser()

    mainparser.add_argument('filename', nargs='+',
                            help='List of file names to be concated')

    arg = mainparser.parse_args()
    return (arg)


if __name__ == '__main__':
    partTraj()


# time    pId     coor.x  coor.y  coor.z  vel.x   vel.y   vel.z   stime   etime   dia     den     partStat
# 3.175031e+01    0       1.002787e+00    9.073364e-01    1.759363e+00    1.488854e-02    -7.370685e-04   5.955675e-04    0.000000e+00    3.175031e+01    1.000000e-07    9.770000e+02    0
