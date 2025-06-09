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


def savePntstec():
    if DEBUG:
        import pdb
        pdb.set_trace()

    args = myparser()

    cnt = 0

    for finame in args.filename:
        print('Reading file {}'.format(finame))

        with open(finame, 'r') as fi:
            for buff in fi:

                # Make new file name and open file.
                cnt += 1
                foname = 'SavePoint{:03d}.dat'.format(cnt)
                with open(foname, 'w') as fo:
                    fo.write('Variables =     X,      Y,      Z\n')
                    fo.write('Zone I = 1,     J = 1,      DATAPACKING = POINT\n')

                    mres = re.match('([0-9e+-.]+)\s+([0-9e+-.]+)\s+([0-9e+-.]+)\s+', buff)
                    fo.write('{}\t{}\t{}\n'.format(mres.group(1), mres.group(2), mres.group(3)))

    
def myparser():
    mainparser = argparse.ArgumentParser()

    mainparser.add_argument('filename', nargs='+',
                            help='List of file names to be concated')

    arg = mainparser.parse_args()
    return (arg)


if __name__ == '__main__':
    savePntstec()


# time    pId     coor.x  coor.y  coor.z  vel.x   vel.y   vel.z   stime   etime   dia     den     partStat
# 3.175031e+01    0       1.002787e+00    9.073364e-01    1.759363e+00    1.488854e-02    -7.370685e-04   5.955675e-04    0.000000e+00    3.175031e+01    1.000000e-07    9.770000e+02    0
