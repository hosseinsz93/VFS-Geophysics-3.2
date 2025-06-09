#ifndef __TEC_OUTPUT_H__
#define __TEC_OUTPUT_H__

#include "variables.h"

void TECOutVec(char const * fname, UserCtx * user, double ***vec);

void TECOutCmpnts(char const * fname, UserCtx * user, Cmpnts ***ftmprt);

void TECOutVecRank(char const * fname, char const * vname, UserCtx * user, double ***vec);

#endif
