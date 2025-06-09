#ifndef __MESH_PARMS_H__
#define __MESH_PARMS_H__

#include "variables.h"


// Set the mesh variables used in most routines.
// to call:
//  PetscInt mx, my, mz; // Dimensions in three directions
//  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
//  PetscInt lxs, lxe, lys, lye, lzs, lze;
//  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
//                  lxs, lxe, lys, lye, lzs, lze);
void meshParms(UserCtx * user,
               PetscInt & mx,  PetscInt & my, PetscInt & mz,
               PetscInt & xs,  PetscInt & xe,
               PetscInt & ys,  PetscInt & ye,
               PetscInt & zs,  PetscInt & ze,
               PetscInt & lxs, PetscInt & lxe,
               PetscInt & lys, PetscInt & lye,
               PetscInt & lzs, PetscInt & lze);


#endif
