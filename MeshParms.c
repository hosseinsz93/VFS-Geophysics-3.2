#include "MeshParms.h"

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
               PetscInt & lzs, PetscInt & lze)
{
  DMDALocalInfo info;
  DMDAGetLocalInfo(user->da, &info);

  mx = info.mx; my = info.my; mz = info.mz;
  xs = info.xs; xe = xs + info.xm;
  ys = info.ys; ye = ys + info.ym;
  zs = info.zs; ze = zs + info.zm;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;
}
