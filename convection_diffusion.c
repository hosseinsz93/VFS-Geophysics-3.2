#include "variables.h"
#include "MeshParms.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>

extern PetscInt immersed, NumberOfBodies, ti, tistart, wallfunction;
extern PetscErrorCode MyKSPMonitor1(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy);
extern PetscInt inletprofile;
extern PetscInt conv_diff, zero_grad;
extern PetscInt SuspendedParticles; //  0: dye or contaminant with no settling velocity; 1: suspended sediment and/or any other particles nd materials with settling velocity
extern PetscReal w_s,U_Bulk,d50,deltab,Background_Conc,Inlet_Conc,sed_density;
extern PetscInt mobile_bed;

double sigma_phi = 1.0;

extern double kappa;

#define PI 3.14159265

double convCtoK(double C) { return C + 273.15; }

void RHS_Conv_Diff(UserCtx *user, Vec ConvDiff_RHS)
{
  DM            da = user->da, fda = user->fda;
  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions
  PetscInt      i, j, k;

  PetscReal     ***aj;

  PetscInt      lxs, lxe, lys, lye, lzs, lze;

  PetscReal     ***Conc, ***Conc_o, ***convdiff_rhs;
  Cmpnts        ***ucont, ***ucat;
  Cmpnts        ***csi, ***eta, ***zet;
  PetscReal     ***nvert, ***lnu_t;

  Vec Fp1, Fp2, Fp3;
  Vec Visc1, Visc2, Visc3;
  PetscReal ***fp1, ***fp2, ***fp3;
  PetscReal ***visc1, ***visc2, ***visc3;

  Cmpnts        ***icsi, ***ieta, ***izet;
  Cmpnts        ***jcsi, ***jeta, ***jzet;
  Cmpnts        ***kcsi, ***keta, ***kzet;
  PetscReal     ***iaj, ***jaj, ***kaj, ***rho, ***mu;


  DMDAGetLocalInfo(da, &info);
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

  VecDuplicate(user->lConc, &Fp1);
  VecDuplicate(user->lConc, &Fp2);
  VecDuplicate(user->lConc, &Fp3);
  VecDuplicate(user->lConc, &Visc1);
  VecDuplicate(user->lConc, &Visc2);
  VecDuplicate(user->lConc, &Visc3);

  VecSet(Fp1,0);
  VecSet(Fp2,0);
  VecSet(Fp3,0);
  VecSet(Visc1,0);
  VecSet(Visc2,0);
  VecSet(Visc3,0);


  if(levelset) {
    DMDAVecGetArray(da, user->lDensity, &rho);
    DMDAVecGetArray(da, user->lMu, &mu);
  }

  if(rans || les) DMDAVecGetArray(da, user->lNu_t, &lnu_t);
  DMDAVecGetArray(fda, user->lUcont, &ucont);
  DMDAVecGetArray(fda, user->lUcat,  &ucat);

  DMDAVecGetArray(fda, user->lCsi, &csi);
  DMDAVecGetArray(fda, user->lEta, &eta);
  DMDAVecGetArray(fda, user->lZet, &zet);

  DMDAVecGetArray(fda, user->lICsi, &icsi);
  DMDAVecGetArray(fda, user->lIEta, &ieta);
  DMDAVecGetArray(fda, user->lIZet, &izet);
  DMDAVecGetArray(fda, user->lJCsi, &jcsi);
  DMDAVecGetArray(fda, user->lJEta, &jeta);
  DMDAVecGetArray(fda, user->lJZet, &jzet);
  DMDAVecGetArray(fda, user->lKCsi, &kcsi);
  DMDAVecGetArray(fda, user->lKEta, &keta);
  DMDAVecGetArray(fda, user->lKZet, &kzet);

  DMDAVecGetArray(da, user->lNvert, &nvert);

  DMDAVecGetArray(da, user->lAj, &aj);
  DMDAVecGetArray(da, user->lIAj, &iaj);
  DMDAVecGetArray(da, user->lJAj, &jaj);
  DMDAVecGetArray(da, user->lKAj, &kaj);

  DMDAVecGetArray(da, user->lConc, &Conc);
  DMDAVecGetArray(da, user->lConc_o, &Conc_o);
  DMDAVecGetArray(da, ConvDiff_RHS, &convdiff_rhs);

  DMDAVecGetArray(da, Fp1, &fp1);
  DMDAVecGetArray(da, Fp2, &fp2);
  DMDAVecGetArray(da, Fp3, &fp3);
  DMDAVecGetArray(da, Visc1, &visc1);
  DMDAVecGetArray(da, Visc2, &visc2);
  DMDAVecGetArray(da, Visc3, &visc3);

  for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
  for (i=lxs-1; i<lxe; i++) {
    double csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
    double eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
    double zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;
    double ajc = iaj[k][j][i];

    double g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
    double g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
    double g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

    double dCdc, dCde, dCdz;
    double nu_t;

    double diff_coef;

    dCdc = Conc[k][j][i+1] - Conc[k][j][i];

    if ((nvert[k][j+1][i])> 1.1 || (nvert[k][j+1][i+1])> 1.1) {
      dCde = (Conc[k][j  ][i+1] + Conc[k][j  ][i] - Conc[k][j-1][i+1] - Conc[k][j-1][i]) * 0.5;
    }
    else if  ((nvert[k][j-1][i])> 1.1 || (nvert[k][j-1][i+1])> 1.1) {
      dCde = (Conc[k][j+1][i+1] + Conc[k][j+1][i] - Conc[k][j  ][i+1] - Conc[k][j  ][i]) * 0.5;
    }
    else {
      dCde = (Conc[k][j+1][i+1] + Conc[k][j+1][i] - Conc[k][j-1][i+1] - Conc[k][j-1][i]) * 0.25;
    }

    if ((nvert[k+1][j][i])> 1.1 || (nvert[k+1][j][i+1])> 1.1) {
      dCdz = (Conc[k  ][j][i+1] + Conc[k  ][j][i] - Conc[k-1][j][i+1] - Conc[k-1][j][i]) * 0.5;
    }
    else if ((nvert[k-1][j][i])> 1.1 || (nvert[k-1][j][i+1])> 1.1) {
      dCdz = (Conc[k+1][j][i+1] + Conc[k+1][j][i] - Conc[k  ][j][i+1] - Conc[k  ][j][i]) * 0.5;
    }
    else {
      dCdz = (Conc[k+1][j][i+1] + Conc[k+1][j][i] - Conc[k-1][j][i+1] - Conc[k-1][j][i]) * 0.25;
    }


    // if( nvert[k][j][i]+nvert[k][j][i+1]>0.1 || i==0 || i==mx-2 || periodic )
    //   fp1[k][j][i] = -ucont[k][j][i].x * Upwind ( Conc[k][j][i], Conc[k][j][i+1], ucont[k][j][i].x);
    // else
    //   fp1[k][j][i] = -ucont[k][j][i].x * weno3 ( Conc[k][j][i-1], Conc[k][j][i], Conc[k][j][i+1], Conc[k][j][i+2], ucont[k][j][i].x );
    fp1[k][j][i] = (nvert[k][j][i]+nvert[k][j][i+1] > 0.5) ? 0.0
                 : -ucont[k][j][i].x * Upwind ( Conc[k][j][i], Conc[k][j][i+1], ucont[k][j][i].x);



    nu_t = 0.0;
    if(rans || les)
    {
      nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j][i+1]);
      nu_t = PetscMax(nu_t, 0);
    }

    double nu = 1./user->ren;
    if (levelset)
    {
      if (nvert[k][j][i] > 0.1 || i==0)
        nu = mu[k][j][i+1];
      else if (nvert[k][j][i+1]>0.1 || i==mx-2)
        nu = mu[k][j][i];
      else
        nu = 0.5 * (mu[k][j][i] + mu[k][j][i+1]);
    }

    if (user->ctemp)
      nu = 1. / (user->ren * user->Pr);

    diff_coef = nu + sigma_phi * nu_t;
    //if (density_current == 2) diff_coef = 0.00000001;
    //if(rans || les) {visc1[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * (nu + sigma_phi * nu_t);}
    //else {visc1[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * (nu + sigma_phi * nu_t);}
    visc1[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * diff_coef;
  }


  for (k=lzs; k<lze; k++)
  for (j=lys-1; j<lye; j++)
  for (i=lxs; i<lxe; i++) {
    double csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
    double eta0= jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
    double zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;
    double ajc = jaj[k][j][i];

    double g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
    double g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
    double g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

    double dCdc, dCde, dCdz;
    double nu_t;

    double diff_coef;


    if ((nvert[k][j][i+1])> 1.1 || (nvert[k][j+1][i+1])> 1.1) {
      dCdc = (Conc[k][j+1][i  ]+ Conc[k][j][i  ] - Conc[k][j+1][i-1] - Conc[k][j][i-1]) * 0.5;
    }
    else if ((nvert[k][j][i-1])> 1.1 || (nvert[k][j+1][i-1])> 1.1) {
      dCdc = (Conc[k][j+1][i+1] + Conc[k][j][i+1] - Conc[k][j+1][i  ] - Conc[k][j][i  ]) * 0.5;
    }
    else {
      dCdc = (Conc[k][j+1][i+1] + Conc[k][j][i+1] - Conc[k][j+1][i-1] - Conc[k][j][i-1]) * 0.25;
    }

    dCde = Conc[k][j+1][i] - Conc[k][j][i];

    if ((nvert[k+1][j][i])> 1.1 || (nvert[k+1][j+1][i])> 1.1) {
      dCdz = (Conc[k  ][j+1][i] + Conc[k  ][j][i] - Conc[k-1][j+1][i] - Conc[k-1][j][i]) * 0.5;
    }
    else if ((nvert[k-1][j][i])> 1.1 || (nvert[k-1][j+1][i])> 1.1) {
      dCdz = (Conc[k+1][j+1][i] + Conc[k+1][j][i] - Conc[k  ][j+1][i] - Conc[k  ][j][i]) * 0.5;
    }
    else {
      dCdz = (Conc[k+1][j+1][i] + Conc[k+1][j][i] - Conc[k-1][j+1][i] - Conc[k-1][j][i]) * 0.25;
    }


    // if( nvert[k][j][i]+nvert[k][j+1][i]>0.1 || j==0 || j==my-2 || periodic )
    //   fp2[k][j][i] = -ucont[k][j][i].y * Upwind ( Conc[k][j][i], Conc[k][j+1][i], ucont[k][j][i].y);
    // else
    //   fp2[k][j][i] = -ucont[k][j][i].y * weno3 ( Conc[k][j-1][i], Conc[k][j][i], Conc[k][j+1][i], Conc[k][j+2][i], ucont[k][j][i].y );
    fp2[k][j][i] = (nvert[k][j][i]+nvert[k][j+1][i] > 0.5) ? 0.0
                 : -ucont[k][j][i].y * Upwind ( Conc[k][j][i], Conc[k][j+1][i], ucont[k][j][i].y);


    nu_t = 0.0;
    if(rans || les)
    {
      nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j+1][i]);
      nu_t = PetscMax(nu_t, 0);
    }

    double nu = 1./user->ren;
    if (levelset) 
    {
      if (nvert[k][j][i] > 0.1 || j==0)
        nu=mu[k][j+1][i];
      else if (nvert[k][j+1][i] > 0.1 || j==my-2)
        nu=mu[k][j][i];
      else
        nu = 0.5 * ( mu[k][j][i] + mu[k][j+1][i] );
    }

    if (user->ctemp)
      nu = 1. / (user->ren * user->Pr);

    diff_coef = nu + sigma_phi * nu_t;
    //if (density_current == 2) diff_coef = 0.00000001;
    //if(rans || les) {visc2[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * (nu + sigma_phi * nu_t);}
    //else {visc2[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * (nu + sigma_phi * nu_t);}
    visc2[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * diff_coef;
  }



  for (k=lzs-1; k<lze; k++)
  for (j=lys; j<lye; j++)
  for (i=lxs; i<lxe; i++) {
    double csi0 = kcsi[k][j][i].x, csi1 = kcsi[k][j][i].y, csi2 = kcsi[k][j][i].z;
    double eta0 = keta[k][j][i].x, eta1 = keta[k][j][i].y, eta2 = keta[k][j][i].z;
    double zet0 = kzet[k][j][i].x, zet1 = kzet[k][j][i].y, zet2 = kzet[k][j][i].z;
    double ajc = kaj[k][j][i];

    double g11 = csi0 * zet0 + csi1 * zet1 + csi2 * zet2;
    double g21 = eta0 * zet0 + eta1 * zet1 + eta2 * zet2;
    double g31 = zet0 * zet0 + zet1 * zet1 + zet2 * zet2;

    double dCdc, dCde, dCdz;
    double nu_t;

    double diff_coef;

    if ((nvert[k][j][i+1])> 1.1 || (nvert[k+1][j][i+1])> 1.1) {
      dCdc = (Conc[k+1][j][i  ] + Conc[k][j][i  ] - Conc[k+1][j][i-1] - Conc[k][j][i-1]) * 0.5;
    }
    else if ((nvert[k][j][i-1])> 1.1 || (nvert[k+1][j][i-1])> 1.1) {
      dCdc = (Conc[k+1][j][i+1] + Conc[k][j][i+1] - Conc[k+1][j][i  ] - Conc[k][j][i  ]) * 0.5;
    }
    else {
      dCdc = (Conc[k+1][j][i+1] + Conc[k][j][i+1] - Conc[k+1][j][i-1] - Conc[k][j][i-1]) * 0.25;
    }

    if ((nvert[k][j+1][i])> 1.1 || (nvert[k+1][j+1][i])> 1.1) {
      dCde = (Conc[k+1][j  ][i] + Conc[k][j  ][i] - Conc[k+1][j-1][i] - Conc[k][j-1][i]) * 0.5;
    }
    else if ((nvert[k][j-1][i])> 1.1 || (nvert[k+1][j-1][i])> 1.1) {
      dCde = (Conc[k+1][j+1][i] + Conc[k][j+1][i] - Conc[k+1][j  ][i] - Conc[k][j  ][i]) * 0.5;
    }
    else {
      dCde = (Conc[k+1][j+1][i] + Conc[k][j+1][i] - Conc[k+1][j-1][i] - Conc[k][j-1][i]) * 0.25;
    }

    dCdz = Conc[k+1][j][i] - Conc[k][j][i];


    ///if( nvert[k][j][i]+nvert[k+1][j][i]>0.1 || k==0 || k==mz-2 || periodic )
    ///  fp3[k][j][i] = -ucont[k][j][i].z * Upwind ( Conc[k][j][i], Conc[k+1][j][i], ucont[k][j][i].z);
    ///else
    ///  fp3[k][j][i] = -ucont[k][j][i].z * weno3 ( Conc[k-1][j][i], Conc[k][j][i], Conc[k+1][j][i], Conc[k+2][j][i], ucont[k][j][i].z );
    fp3[k][j][i] = (nvert[k][j][i]+nvert[k+1][j][i] > 0.5) ? 0.0
                 : -ucont[k][j][i].z * Upwind ( Conc[k][j][i], Conc[k+1][j][i], ucont[k][j][i].z);

    if (rans || les)
    {
      nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k+1][j][i]);
      nu_t = PetscMax(nu_t, 0);
    }

    double nu = 1./user->ren;
    if (levelset)
    {
      if (nvert[k][j][i] > 0.1 || k==0)
        nu=mu[k+1][j][i];
      else if (nvert[k+1][j][i] > 0.1 || k==mz-2)
        nu=mu[k][j][i];
      else nu = 0.5 * ( mu[k][j][i] + mu[k+1][j][i] );
    }

    if (user->ctemp)
      nu = 1. / (user->ren * user->Pr);

    diff_coef = nu + sigma_phi * nu_t;
    //if (density_current ==2) diff_coef = 0.00000001;
    //if(rans || les) {visc3[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * (nu + sigma_phi * nu_t);}
    //else {visc3[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * (nu + sigma_phi * nu_t);}
    visc3[k][j][i] = (g11 * dCdc + g21 * dCde + g31 * dCdz) * ajc * diff_coef;
  }


  DMDAVecRestoreArray(da, Fp1, &fp1);
  DMDAVecRestoreArray(da, Fp2, &fp2);
  DMDAVecRestoreArray(da, Fp3, &fp3);
  DMDAVecRestoreArray(da, Visc1, &visc1);
  DMDAVecRestoreArray(da, Visc2, &visc2);
  DMDAVecRestoreArray(da, Visc3, &visc3);

  DMDALocalToLocalBegin(da, Fp1, INSERT_VALUES, Fp1);
  DMDALocalToLocalEnd(da, Fp1, INSERT_VALUES, Fp1);

  DMDALocalToLocalBegin(da, Fp2, INSERT_VALUES, Fp2);
  DMDALocalToLocalEnd(da, Fp2, INSERT_VALUES, Fp2);

  DMDALocalToLocalBegin(da, Fp3, INSERT_VALUES, Fp3);
  DMDALocalToLocalEnd(da, Fp3, INSERT_VALUES, Fp3);

  DMDALocalToLocalBegin(da, Visc1, INSERT_VALUES, Visc1);
  DMDALocalToLocalEnd(da, Visc1, INSERT_VALUES, Visc1);

  DMDALocalToLocalBegin(da, Visc2, INSERT_VALUES, Visc2);
  DMDALocalToLocalEnd(da, Visc2, INSERT_VALUES, Visc2);

  DMDALocalToLocalBegin(da, Visc3, INSERT_VALUES, Visc3);
  DMDALocalToLocalEnd(da, Visc3, INSERT_VALUES, Visc3);

  DMDAVecGetArray(da, Fp1, &fp1);
  DMDAVecGetArray(da, Fp2, &fp2);
  DMDAVecGetArray(da, Fp3, &fp3);
  DMDAVecGetArray(da, Visc1, &visc1);
  DMDAVecGetArray(da, Visc2, &visc2);
  DMDAVecGetArray(da, Visc3, &visc3);

  if (periodic)
    for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
    for (i=xs; i<xe; i++) {
      int a=i, b=j, c=k;

      int flag=0;

      if(i_periodic && i==0) a=mx-2, flag=1;
      else if(i_periodic && i==mx-1) a=1, flag=1;

      if(j_periodic && j==0) b=my-2, flag=1;
      else if(j_periodic && j==my-1) b=1, flag=1;

      if(k_periodic && k==0) c=mz-2, flag=1;
      else if(k_periodic && k==mz-1) c=1, flag=1;

      if(ii_periodic && i==0) a=-2, flag=1;
      else if(ii_periodic && i==mx-1) a=mx+1, flag=1;

      if(jj_periodic && j==0) b=-2, flag=1;
      else if(jj_periodic && j==my-1) b=my+1, flag=1;

      if(kk_periodic && k==0) c=-2, flag=1;
      else if(kk_periodic && k==mz-1) c=mz+1, flag=1;

      if(flag) {
        fp1[k][j][i] = fp1[c][b][a];
        fp2[k][j][i] = fp2[c][b][a];
        fp3[k][j][i] = fp3[c][b][a];
        visc1[k][j][i] = visc1[c][b][a];
        visc2[k][j][i] = visc2[c][b][a];
        visc3[k][j][i] = visc3[c][b][a];
      }
    }



  for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
  for (i=lxs; i<lxe; i++) {
    if ( i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 || nvert[k][j][i]>0.1 ) { // it was 0.1 for k-omega  not 1.1
      convdiff_rhs[k][j][i] = 0;
      //komega_rhs[k][j][i].x = komega_rhs[k][j][i].y = 0;
      continue;
    }

    double ajc = aj[k][j][i];

    if ( nvert[k][j][i] < 0.1 ) {
      double r = 1.;

      if(levelset) r = rho[k][j][i];

      // advection
      convdiff_rhs[k][j][i]   = (fp1[k][j][i] - fp1[k][j][i-1]
                               + fp2[k][j][i] - fp2[k][j-1][i]
                               + fp3[k][j][i] - fp3[k-1][j][i] ) * ajc;

      // diffusion
      convdiff_rhs[k][j][i] += (visc1[k][j][i] - visc1[k][j][i-1]
                              + visc2[k][j][i] - visc2[k][j-1][i]
                              + visc3[k][j][i] - visc3[k-1][j][i]) * ajc / r;
    }
  }

  if(rans || les) DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);

  DMDAVecRestoreArray(da, Fp1, &fp1);
  DMDAVecRestoreArray(da, Fp2, &fp2);
  DMDAVecRestoreArray(da, Fp3, &fp3);
  DMDAVecRestoreArray(da, Visc1, &visc1);
  DMDAVecRestoreArray(da, Visc2, &visc2);
  DMDAVecRestoreArray(da, Visc3, &visc3);

  DMDAVecRestoreArray(fda, user->lUcont, &ucont);
  DMDAVecRestoreArray(fda, user->lUcat,  &ucat);

  DMDAVecRestoreArray(fda, user->lCsi, &csi);
  DMDAVecRestoreArray(fda, user->lEta, &eta);
  DMDAVecRestoreArray(fda, user->lZet, &zet);

  DMDAVecRestoreArray(fda, user->lICsi, &icsi);
  DMDAVecRestoreArray(fda, user->lIEta, &ieta);
  DMDAVecRestoreArray(fda, user->lIZet, &izet);
  DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
  DMDAVecRestoreArray(fda, user->lJEta, &jeta);
  DMDAVecRestoreArray(fda, user->lJZet, &jzet);
  DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
  DMDAVecRestoreArray(fda, user->lKEta, &keta);
  DMDAVecRestoreArray(fda, user->lKZet, &kzet);

  DMDAVecRestoreArray(da, user->lNvert, &nvert);

  DMDAVecRestoreArray(da, user->lAj, &aj);
  DMDAVecRestoreArray(da, user->lIAj, &iaj);
  DMDAVecRestoreArray(da, user->lJAj, &jaj);
  DMDAVecRestoreArray(da, user->lKAj, &kaj);

  DMDAVecRestoreArray(da, user->lConc, &Conc);
  DMDAVecRestoreArray(da, user->lConc_o, &Conc_o);
  DMDAVecRestoreArray(da, ConvDiff_RHS, &convdiff_rhs);

  if(levelset) {
    DMDAVecRestoreArray(da, user->lDensity, &rho);
    DMDAVecRestoreArray(da, user->lMu, &mu);
  }

  VecDestroy(&Fp1);
  VecDestroy(&Fp2);
  VecDestroy(&Fp3);
  VecDestroy(&Visc1);
  VecDestroy(&Visc2);
  VecDestroy(&Visc3);
};


void Force_Current(UserCtx *user)
{
  DM            da = user->da;
  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions
  PetscInt      i, j, k;
  double Ri = user->Ri;


  PetscInt      lxs, lxe, lys, lye, lzs, lze;


  PetscReal     ***conc;
  PetscReal     ***nvert;

  Cmpnts                ***fcurrent, ***csi, ***eta, ***zet;

  DMDAGetLocalInfo(da, &info);
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

  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(user->da, user->lConc, &conc);
  DMDAVecGetArray(user->fda, user->lFCurrent, &fcurrent);
  DMDAVecGetArray(user->fda, user->lCsi,  &csi);
  DMDAVecGetArray(user->fda, user->lEta,  &eta);
  DMDAVecGetArray(user->fda, user->lZet,  &zet);


  double force_x, force_y, force_z;

  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++) { // pressure node; cell center
    force_x = 0.0;
    force_y = 0.0;
    force_z = 0.0;

    // if (user->bctype_tmprt[0] || user->bctype_tmprt[1]) force_x = Ri * (tmprt[k][j][i]-tmprt_xzAvg[j]);
    // if (user->bctype_tmprt[2] || user->bctype_tmprt[3]) force_y = Ri * tmprt[k][j][i];
    // if (user->bctype_tmprt[4] || user->bctype_tmprt[5]) force_z = Ri * (tmprt[k][j][i]-tmprt_xzAvg[j]);

    if (user->ctemp)
    {
      static int outputMess = false;

      if (user->ctemp == 1)
      {
        force_z = /*force_ri =*/ user->Ri_z * (conc[k][j][i] - user->buoyTempRef)
                              / (user->buoyTemp - user->buoyTempRef);
        if (!outputMess)
          PetscPrintf(PETSC_COMM_WORLD, "THE BOUYANCY FORCE IS CALCUATED USING A RICHARDSON NUMBER OF %f.\n", user->Ri_z);
      }
      else if (user->ctemp == 2)
      {
        // Boussinesq approximation - type 1 - option 2
        force_z = /*force_ba1 =*/ (1.0 - user->buoyBeta * (user->buoyTempRef - conc[k][j][i])) * gravity_z;
        if (!outputMess)
          PetscPrintf(PETSC_COMM_WORLD, "THE BOUYANCY FORCE IS CALCULATED USING THE FIRST IMPLEMENTATION OF THE BOUSSINESQ APPROXIMATION.\n");
      }
      else if (user->ctemp == 3)
      {
        // Boussinesq approximation - type 2 - option 3
        force_z = /*force_ba2 =*/ user->buoyBeta * (conc[k][j][i] - user->buoyTempRef) * gravity_z;
        if (!outputMess)
          PetscPrintf(PETSC_COMM_WORLD, "THE BOUYANCY FORCE IS CALCULATED USING THE SECOND IMPLEMENTATION OF THE BOUSSINESQ APPROXIMATION.\n");
      }
      else if (user->ctemp == 4)
      {
        // Reduced gravity - option 4
        force_z = /*force_rg* =*/ gravity_z * (convCtoK(conc[k][j][i]) / convCtoK(user->buoyTempRef) - 1.0);
        if (!outputMess)
          PetscPrintf(PETSC_COMM_WORLD, "THE BOUYANCY FORCE IS CALCUATED USING THE REDUCED GRAVITY METHOD.\n");
      }
      else
        if (!outputMess)
          PetscPrintf(PETSC_COMM_WORLD, "NO BUOYANCY FORCE IS CALCULATED SO THE VALUE IS ZERO.\n");

      outputMess = true;
    }

    fcurrent[k][j][i].x = force_x *  csi[k][j][i].x + force_y *  csi[k][j][i].y + force_z *  csi[k][j][i].z;
    fcurrent[k][j][i].y = force_x *  eta[k][j][i].x + force_y *  eta[k][j][i].y + force_z *  eta[k][j][i].z;
    fcurrent[k][j][i].z = force_x *  zet[k][j][i].x + force_y *  zet[k][j][i].y + force_z *  zet[k][j][i].z;

    // if (k==3 && i==3) printf("the force  %d %d %le %le %le %le\n", my-2, j, ftmprt[k][j][i].y, force_x, force_y, force_z);



    if( nvert[k][j][i] > 0.5)
    {
      fcurrent[k][j][i].x = 0.0;
      fcurrent[k][j][i].y = 0.0;
      fcurrent[k][j][i].z = 0.0;
    }
    else if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 || j==my-2) {
      fcurrent[k][j][i].x = 0.0;
      fcurrent[k][j][i].y = 0.0;
      fcurrent[k][j][i].z = 0.0;

    }

    // if (user->bctype_tmprt[1] && i==mx-2) ftmprt[k][j][i].x = 0.0;
    // if (user->bctype_tmprt[3] && j==my-2) ftmprt[k][j][i].y = 0.0;
    // if (user->bctype_tmprt[5] && k==mz-2) ftmprt[k][j][i].z = 0.0;


  }

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, user->lConc, &conc);
  DMDAVecRestoreArray(user->fda, user->lFCurrent, &fcurrent);

  DMLocalToGlobalBegin(user->fda, user->lFCurrent, INSERT_VALUES, user->FCurrent);
  DMLocalToGlobalEnd(user->fda, user->lFCurrent, INSERT_VALUES, user->FCurrent);


  // TECIOOut_rhs(user, user->FTmprt);

  DMDAVecRestoreArray(user->fda, user->lCsi,  &csi);
  DMDAVecRestoreArray(user->fda, user->lEta,  &eta);
  DMDAVecRestoreArray(user->fda, user->lZet,  &zet);

};

void Conv_Diff_IC(UserCtx *user)
{
  DM            da = user->da, fda = user->fda;
  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions
  PetscInt      i, j, k;
  Vec Coor;
  Cmpnts        ***coor;

  double nu = 1./user->ren;

  PetscInt      lxs, lxe, lys, lye, lzs, lze;


  PetscReal     ***nvert;
  PetscReal     ***Conc;

  DMDAGetLocalInfo(da, &info);
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

  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(da, user->lConc, &Conc);

  // DMDAGetCoordinates(da, &Coor);
  // DMDAVecGetArray(user->fda, Coor, &coor);
  DMDAGetGhostedCoordinates(da, &Coor);
  DMDAVecGetArray(fda, Coor, &coor);

  // I.C. for Conc
  for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
  for (i=lxs; i<lxe; i++) { // concentration node; cell center

    // Initial conditions for mouth/nose temperature calculation.
    if (user->ctemp)
    {
      Conc[k][j][i] = (nvert[k][j][i] < 1.1) ? user->buoyTempRef : 0.0;
    }
    else
    {
      //Conc[k][j][i] = 0.55;
    }
    if(density_current == 1){
      double xc  = (coor[k][j][i].x + coor[k][j-1][i].x + coor[k][j][i-1].x + coor[k][j-1][i-1].x) * 0.25 ;
      if (xc < 1. && nvert[k][j][i]<1.1) {
        Conc[k][j][i]=1.0;
        if(j==my-2) Conc[k][j+1][i]=1.0;
        if(j==1) Conc[k][j-1][i]=1.0;
        if(i==mx-2) Conc[k][j][i+1]=1.0;
        if(i==1) Conc[k][j][i-1]=1.0;
        if(k==1) Conc[k-1][j][i]=1.0;
      }
      else   {
        Conc[k][j][i]=0.0;
        if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ) Conc[k][j][i] = 0.;
      }
    }
    else if(density_current == 2) {
      if( nvert[k][j][i]<5.1) Conc[k][j][i] = 0.0;
    }
    else if(density_current == 3) {
      if( nvert[k][j][i]<5.1) Conc[k][j][i] = 0.0;
    }
    else {
      if( nvert[k][j][i]<5.1) Conc[k][j][i] = 0.0;
    }

    if( nvert[k][j][i]>1.5) {   //Solid nodes common command for all
      Conc[k][j][i] = 0.0;
    }
    else if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ) {   //commom command for all
      Conc[k][j][i] = 0.;
    }

    if( nvert[k][j][i]<= 1.2) {   //fluid node and ib
      Conc[k][j][i] = Background_Conc;
    }
  }

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(da, user->lConc, &Conc);
  DMDAVecRestoreArray(fda, Coor, &coor);

  DMLocalToGlobalBegin(da, user->lConc, INSERT_VALUES, user->Conc);
  DMLocalToGlobalEnd(da, user->lConc, INSERT_VALUES, user->Conc);

  VecCopy(user->Conc, user->Conc_o);

};

void SettlingVelocityCorrection (UserCtx *user)
{
  PetscInt      i, j, k;
  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze;
  PetscInt      mx,my,mz;
  PetscInt      lxs, lxe, lys, lye, lzs, lze;
  Cmpnts        ***ucont;
  PetscReal     ***nvert;

  DM            da = user->da,fda = user->fda;
  info = user->info;

  xs = info.xs; xe = info.xs + info.xm;
  ys = info.ys; ye = info.ys + info.ym;
  zs = info.zs; ze = info.zs + info.zm;
  mx = info.mx; my = info.my; mz = info.mz;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  Cmpnts ***ucat, ***icsi, ***jeta, ***kzet;

  /*
    if(!immersed && ti==tistart) {
      DAVecGetArray(da, user->lNvert, &nvert);
      for (k=lzs; k<lze; k++)
      for (j=lys; j<lye; j++)
      for (i=lxs; i<lxe; i++) {
        if(user->bctype[0]==-1 && i==1) nvert[k][j][i]=1;
        if(user->bctype[1]==-1 && i==mx-2) nvert[k][j][i]=1;
        if(user->bctype[2]==-1 && j==1) nvert[k][j][i]=1;
        if(user->bctype[3]==-1 && j==my-2) nvert[k][j][i]=1;
        if(user->bctype[4]==-1 && k==1) nvert[k][j][i]=1;
        if(user->bctype[5]==-1 && k==mz-2) nvert[k][j][i]=1;

        if(user->bctype[0]==-2 && i==1) nvert[k][j][i]=1;
        if(user->bctype[1]==-2 && i==mx-2) nvert[k][j][i]=1;
        if(user->bctype[2]==-2 && j==1) nvert[k][j][i]=1;
        if(user->bctype[3]==-2 && j==my-2) nvert[k][j][i]=1;
        if(user->bctype[4]==-2 && k==1) nvert[k][j][i]=1;
        if(user->bctype[5]==-2 && k==mz-2) nvert[k][j][i]=1;
      }
      DAVecRestoreArray(da, user->lNvert, &nvert);
      DALocalToLocalBegin(da, user->lNvert, INSERT_VALUES, user->lNvert);
      DALocalToLocalEnd(da, user->lNvert, INSERT_VALUES, user->lNvert);
      DALocalToGlobal(da, user->lNvert, INSERT_VALUES, user->Nvert);
    }
  */

  DMDAVecGetArray(fda, user->lUcat, &ucat);
  DMDAVecGetArray(fda, user->lUcont, &ucont);
  DMDAVecGetArray(fda, user->lICsi, &icsi);
  DMDAVecGetArray(fda, user->lJEta, &jeta);
  DMDAVecGetArray(fda, user->lKZet, &kzet);
  DMDAVecGetArray(da, user->lNvert, &nvert);

  /*
    if (periodic)
      for (k=zs; k<ze; k++)
      for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++) {
        int flag=0, a=i, b=j, c=k;

        if(i_periodic && i==0) a=mx-2, flag=1;
        else if(i_periodic && i==mx-1) a=1, flag=1;

        if(j_periodic && j==0) b=my-2, flag=1;
        else if(j_periodic && j==my-1) b=1, flag=1;

        if(k_periodic && k==0) c=mz-2, flag=1;
        else if(k_periodic && k==mz-1) c=1, flag=1;

        if(ii_periodic && i==0) a=-2, flag=1;
        else if(ii_periodic && i==mx-1) a=mx+1, flag=1;

        if(jj_periodic && j==0) b=-2, flag=1;
        else if(jj_periodic && j==my-1) b=my+1, flag=1;

        if(kk_periodic && k==0) c=-2, flag=1;
        else if(kk_periodic && k==mz-1) c=mz+1, flag=1;

        if(flag) {
          ucat[k][j][i] = ucat[c][b][a];
        }
      }
  */

  double  ucx, ucy, ucz;


  // DEBUG: is settling velocity being applied.
  // PetscPrintf(PETSC_COMM_WORLD, "In SettlingVelocityCorrection, w_s = %f\n", w_s);
  for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
  for (i=lxs; i<lxe; i++) {

    //if(immersed) {
    //double f = 1.0;

    //if(immersed==3) f = 0;

    // if ((int)nvert[k][j][i]==1) {
    ucx = (ucat[k][j][i].x + ucat[k][j][i+1].x) * 0.5;
    ucy = (ucat[k][j][i].y + ucat[k][j][i+1].y) * 0.5;
    ucz = (ucat[k][j][i].z + ucat[k][j][i+1].z) * 0.5 - w_s;
    ucont[k][j][i].x = (ucx * icsi[k][j][i].x + ucy * icsi[k][j][i].y + ucz * icsi[k][j][i].z);

    ucx = (ucat[k][j+1][i].x + ucat[k][j][i].x) * 0.5;
    ucy = (ucat[k][j+1][i].y + ucat[k][j][i].y) * 0.5;
    ucz = (ucat[k][j+1][i].z + ucat[k][j][i].z) * 0.5 - w_s;
    ucont[k][j][i].y = (ucx * jeta[k][j][i].x + ucy * jeta[k][j][i].y + ucz * jeta[k][j][i].z);

    ucx = (ucat[k+1][j][i].x + ucat[k][j][i].x) * 0.5;
    ucy = (ucat[k+1][j][i].y + ucat[k][j][i].y) * 0.5;
    ucz = (ucat[k+1][j][i].z + ucat[k][j][i].z) * 0.5 - w_s;
    ucont[k][j][i].z = (ucx * kzet[k][j][i].x + ucy * kzet[k][j][i].y + ucz * kzet[k][j][i].z);
    // }

    /*
      if ((int)(nvert[k][j][i+1])==1) {
        ucx = (ucat[k][j][i].x + ucat[k][j][i+1].x) * 0.5;
        ucy = (ucat[k][j][i].y + ucat[k][j][i+1].y) * 0.5;
        ucz = (ucat[k][j][i].z + ucat[k][j][i+1].z) * 0.5;
        ucont[k][j][i].x = (ucx * icsi[k][j][i].x + ucy * icsi[k][j][i].y + ucz * icsi[k][j][i].z);
      }

      if ((int)(nvert[k][j+1][i])==1) {
        ucx = (ucat[k][j+1][i].x + ucat[k][j][i].x) * 0.5;
        ucy = (ucat[k][j+1][i].y + ucat[k][j][i].y) * 0.5;
        ucz = (ucat[k][j+1][i].z + ucat[k][j][i].z) * 0.5;
        ucont[k][j][i].y = (ucx * jeta[k][j][i].x + ucy * jeta[k][j][i].y + ucz * jeta[k][j][i].z);
      }

      if ((int)(nvert[k+1][j][i])==1) {
        ucx = (ucat[k+1][j][i].x + ucat[k][j][i].x) * 0.5;
        ucy = (ucat[k+1][j][i].y + ucat[k][j][i].y) * 0.5;
        ucz = (ucat[k+1][j][i].z + ucat[k][j][i].z) * 0.5;
        ucont[k][j][i].z = (ucx * kzet[k][j][i].x + ucy * kzet[k][j][i].y + ucz * kzet[k][j][i].z);
      }

      if(!movefsi && !rotatefsi && immersed==3) {
        if ((nvert[k][j][i+1]+nvert[k][j][i])>1.1) ucont[k][j][i].x = 0;
        if ((nvert[k][j+1][i]+nvert[k][j][i])>1.1) ucont[k][j][i].y = 0;
        if ((nvert[k+1][j][i]+nvert[k][j][i])>1.1) ucont[k][j][i].z = 0;
      }
    */

    // wall func

    if ( (user->bctype[0]==-1 && i==1) ||( user->bctype[1]==-1 && i==mx-2) ) {
      if(i==1) ucont[k][j][i-1].x=0;
      else ucont[k][j][i].x=0;
    }

    if ( (user->bctype[0]==-2 && i==1) ||( user->bctype[1]==-2 && i==mx-2) ) {
      if(i==1) ucont[k][j][i-1].x=0;
      else ucont[k][j][i].x=0;
    }

    if ( (user->bctype[2]==-1 && j==1) || (user->bctype[3]==-1 && j==my-2) ) {
      if(j==1) ucont[k][j-1][i].y=0;
      else ucont[k][j][i].y=0;
    }

    if ( (user->bctype[2]==-2 && j==1) || (user->bctype[3]==-2 && j==my-2) ) {
      if(j==1) ucont[k][j-1][i].y=0;
      else ucont[k][j][i].y=0;
    }

  }

  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++) {
    if ( (user->bctype[0]==10) && i==1) ucont[k][j][0].x = 0;
    if ( (user->bctype[1]==10) && i==mx-2) ucont[k][j][mx-2].x = 0;
    if ( (user->bctype[2]==10) && j==1) ucont[k][0][i].y = 0;
    if ( (user->bctype[3]==10) && j==my-2) ucont[k][my-2][i].y = 0;
    if ( (user->bctype[4]==10) && k==1) ucont[0][j][i].z = 0;
    if ( (user->bctype[5]==10) && k==mz-2) ucont[mz-2][j][i].z = 0;

    if ( i_periodic && i==0 ) ucont[k][j][0].x = ucont[k][j][mx-2].x;
    if ( i_periodic && i==mx-1 ) ucont[k][j][mx-1].x = ucont[k][j][1].x;
    if ( j_periodic && j==0 ) ucont[k][0][i].y = ucont[k][my-2][i].y;
    if ( j_periodic && j==my-1 ) ucont[k][my-1][i].y = ucont[k][1][i].y;
    if ( k_periodic && k==0 ) ucont[0][j][i].z = ucont[mz-2][j][i].z;
    if ( k_periodic && k==mz-1 ) ucont[mz-1][j][i].z = ucont[1][j][i].z;

    if ( ii_periodic && i==0 ) ucont[k][j][0].x = ucont[k][j][-2].x;
    if ( ii_periodic && i==mx-1 ) ucont[k][j][mx-1].x = ucont[k][j][mx+1].x;

    if ( jj_periodic && j==0 ) ucont[k][0][i].y = ucont[k][-2][i].y;
    if ( jj_periodic && j==my-1 ) ucont[k][my-1][i].y = ucont[k][my+1][i].y;

    if ( kk_periodic && k==0 ) ucont[0][j][i].z = ucont[-2][j][i].z;
    if ( kk_periodic && k==mz-1 ) ucont[mz-1][j][i].z = ucont[mz+1][j][i].z;
  }
  DMDAVecRestoreArray(fda, user->lUcat, &ucat);
  DMDAVecRestoreArray(fda, user->lICsi, &icsi);
  DMDAVecRestoreArray(fda, user->lJEta, &jeta);
  DMDAVecRestoreArray(fda, user->lKZet, &kzet);
  DMDAVecRestoreArray(fda, user->lUcont, &ucont);
  DMDAVecRestoreArray(da, user->lNvert, &nvert);


  DMDALocalToLocalBegin(fda, user->lUcont, INSERT_VALUES, user->lUcont);
  DMDALocalToLocalEnd(fda, user->lUcont, INSERT_VALUES, user->lUcont);

  return;
}

void SettlingVelocityCorrectionBack (UserCtx *user)
{
  PetscInt      i, j, k;
  DMDALocalInfo info ;
  PetscInt      xs, xe, ys, ye, zs, ze;
  PetscInt      mx,my,mz;
  PetscInt      lxs, lxe, lys, lye, lzs, lze;
  Cmpnts        ***ucont;
  PetscReal     ***nvert;

  DM            da = user->da,fda = user->fda;
  info = user->info;

  xs = info.xs; xe = info.xs + info.xm;
  ys = info.ys; ye = info.ys + info.ym;
  zs = info.zs; ze = info.zs + info.zm;
  mx = info.mx; my = info.my; mz = info.mz;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  Cmpnts ***ucat, ***icsi, ***jeta, ***kzet;


  DMDAVecGetArray(fda, user->lUcat, &ucat);
  DMDAVecGetArray(fda, user->lUcont, &ucont);
  DMDAVecGetArray(fda, user->lICsi, &icsi);
  DMDAVecGetArray(fda, user->lJEta, &jeta);
  DMDAVecGetArray(fda, user->lKZet, &kzet);
  DMDAVecGetArray(da, user->lNvert, &nvert);

  double  ucx, ucy, ucz;

  for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
  for (i=lxs; i<lxe; i++) {

    //if(immersed) {
    //double f = 1.0;

    //if(immersed==3) f = 0;

    // if ((int)nvert[k][j][i]==1) {
    ucx = (ucat[k][j][i].x + ucat[k][j][i+1].x) * 0.5;
    ucy = (ucat[k][j][i].y + ucat[k][j][i+1].y) * 0.5;
    ucz = (ucat[k][j][i].z + ucat[k][j][i+1].z) * 0.5;
    ucont[k][j][i].x = (ucx * icsi[k][j][i].x + ucy * icsi[k][j][i].y + ucz * icsi[k][j][i].z);

    ucx = (ucat[k][j+1][i].x + ucat[k][j][i].x) * 0.5;
    ucy = (ucat[k][j+1][i].y + ucat[k][j][i].y) * 0.5;
    ucz = (ucat[k][j+1][i].z + ucat[k][j][i].z) * 0.5;
    ucont[k][j][i].y = (ucx * jeta[k][j][i].x + ucy * jeta[k][j][i].y + ucz * jeta[k][j][i].z);

    ucx = (ucat[k+1][j][i].x + ucat[k][j][i].x) * 0.5;
    ucy = (ucat[k+1][j][i].y + ucat[k][j][i].y) * 0.5;
    ucz = (ucat[k+1][j][i].z + ucat[k][j][i].z) * 0.5;
    ucont[k][j][i].z = (ucx * kzet[k][j][i].x + ucy * kzet[k][j][i].y + ucz * kzet[k][j][i].z);
    // }

    /*
      if ((int)(nvert[k][j][i+1])==1) {
        ucx = (ucat[k][j][i].x + ucat[k][j][i+1].x) * 0.5;
        ucy = (ucat[k][j][i].y + ucat[k][j][i+1].y) * 0.5;
        ucz = (ucat[k][j][i].z + ucat[k][j][i+1].z) * 0.5;
        ucont[k][j][i].x = (ucx * icsi[k][j][i].x + ucy * icsi[k][j][i].y + ucz * icsi[k][j][i].z);
      }

      if ((int)(nvert[k][j+1][i])==1) {
        ucx = (ucat[k][j+1][i].x + ucat[k][j][i].x) * 0.5;
        ucy = (ucat[k][j+1][i].y + ucat[k][j][i].y) * 0.5;
        ucz = (ucat[k][j+1][i].z + ucat[k][j][i].z) * 0.5;
        ucont[k][j][i].y = (ucx * jeta[k][j][i].x + ucy * jeta[k][j][i].y + ucz * jeta[k][j][i].z);
      }

      if ((int)(nvert[k+1][j][i])==1) {
        ucx = (ucat[k+1][j][i].x + ucat[k][j][i].x) * 0.5;
        ucy = (ucat[k+1][j][i].y + ucat[k][j][i].y) * 0.5;
        ucz = (ucat[k+1][j][i].z + ucat[k][j][i].z) * 0.5;
        ucont[k][j][i].z = (ucx * kzet[k][j][i].x + ucy * kzet[k][j][i].y + ucz * kzet[k][j][i].z);
      }

      if(!movefsi && !rotatefsi && immersed==3) {
        if ((nvert[k][j][i+1]+nvert[k][j][i])>1.1) ucont[k][j][i].x = 0;
        if ((nvert[k][j+1][i]+nvert[k][j][i])>1.1) ucont[k][j][i].y = 0;
        if ((nvert[k+1][j][i]+nvert[k][j][i])>1.1) ucont[k][j][i].z = 0;
      }
    */

    // wall func

    if ( (user->bctype[0]==-1 && i==1) ||( user->bctype[1]==-1 && i==mx-2) ) {
      if(i==1) ucont[k][j][i-1].x=0;
      else ucont[k][j][i].x=0;
    }

    if ( (user->bctype[0]==-2 && i==1) ||( user->bctype[1]==-2 && i==mx-2) ) {
      if(i==1) ucont[k][j][i-1].x=0;
      else ucont[k][j][i].x=0;
    }

    if ( (user->bctype[2]==-1 && j==1) || (user->bctype[3]==-1 && j==my-2) ) {
      if(j==1) ucont[k][j-1][i].y=0;
      else ucont[k][j][i].y=0;
    }

    if ( (user->bctype[2]==-2 && j==1) || (user->bctype[3]==-2 && j==my-2) ) {
      if(j==1) ucont[k][j-1][i].y=0;
      else ucont[k][j][i].y=0;
    }

  }

  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++) {
    if ( (user->bctype[0]==10) && i==1) ucont[k][j][0].x = 0;
    if ( (user->bctype[1]==10) && i==mx-2) ucont[k][j][mx-2].x = 0;
    if ( (user->bctype[2]==10) && j==1) ucont[k][0][i].y = 0;
    if ( (user->bctype[3]==10) && j==my-2) ucont[k][my-2][i].y = 0;
    if ( (user->bctype[4]==10) && k==1) ucont[0][j][i].z = 0;
    if ( (user->bctype[5]==10) && k==mz-2) ucont[mz-2][j][i].z = 0;

    if ( i_periodic && i==0 ) ucont[k][j][0].x = ucont[k][j][mx-2].x;
    if ( i_periodic && i==mx-1 ) ucont[k][j][mx-1].x = ucont[k][j][1].x;
    if ( j_periodic && j==0 ) ucont[k][0][i].y = ucont[k][my-2][i].y;
    if ( j_periodic && j==my-1 ) ucont[k][my-1][i].y = ucont[k][1][i].y;
    if ( k_periodic && k==0 ) ucont[0][j][i].z = ucont[mz-2][j][i].z;
    if ( k_periodic && k==mz-1 ) ucont[mz-1][j][i].z = ucont[1][j][i].z;

    if ( ii_periodic && i==0 ) ucont[k][j][0].x = ucont[k][j][-2].x;
    if ( ii_periodic && i==mx-1 ) ucont[k][j][mx-1].x = ucont[k][j][mx+1].x;

    if ( jj_periodic && j==0 ) ucont[k][0][i].y = ucont[k][-2][i].y;
    if ( jj_periodic && j==my-1 ) ucont[k][my-1][i].y = ucont[k][my+1][i].y;

    if ( kk_periodic && k==0 ) ucont[0][j][i].z = ucont[-2][j][i].z;
    if ( kk_periodic && k==mz-1 ) ucont[mz-1][j][i].z = ucont[mz+1][j][i].z;
  }
  DMDAVecRestoreArray(fda, user->lUcat, &ucat);
  DMDAVecRestoreArray(fda, user->lICsi, &icsi);
  DMDAVecRestoreArray(fda, user->lJEta, &jeta);
  DMDAVecRestoreArray(fda, user->lKZet, &kzet);
  DMDAVecRestoreArray(fda, user->lUcont, &ucont);
  DMDAVecRestoreArray(da, user->lNvert, &nvert);


  DMDALocalToLocalBegin(fda, user->lUcont, INSERT_VALUES, user->lUcont);
  DMDALocalToLocalEnd(fda, user->lUcont, INSERT_VALUES, user->lUcont);

  return;
}


double Depth_Averaged_Conc( PetscReal ***Conc, PetscReal ***nvert, int i, int j, int k, int lye)
{
/*
  DA            da = user->da, fda = user->fda;
  DALocalInfo   info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions

  PetscReal     ***Conc;
  PetscReal     ***nvert;

  DAGetLocalInfo(da, &info);
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

  DAVecGetArray(da, user->lConc, &Conc);
  DAVecGetArray(da, user->lNvert, &nvert);
*/
  PetscInt      jj;
  double c_v;
  double sum = 0.;
  c_v = 0.;
  for (jj=j-1; jj<lye; jj++)
  {
    if ( nvert[k][jj][i]<1.1) {
      sum += 1.;
      c_v += Conc[k][jj][i];
    }
  }
  if(sum>0.) c_v = c_v/sum;

/*
  DAVecRestoreArray(da, user->lConc, &Conc);
  DALocalToLocalBegin(da, user->lConc, INSERT_VALUES, user->lConc);
  DALocalToLocalEnd(da, user->lConc, INSERT_VALUES, user->lConc);
*/
  return c_v;
}


template < typename T > T sqr(T a) { return a*a; }

void coorMinMax(int i, int j, int k, Cmpnts ***coor,
                double (&minc)[3], double (&maxc)[3])
{
  minc[0] = minc[1] = minc[2] = std::numeric_limits<double>::max();
  maxc[0] = maxc[1] = maxc[2] = std::numeric_limits<double>::min();

  int idx[8][3] = {
    {  0,  0,  0 },
    {  0, -1,  0 },
    { -1,  0,  0 },
    { -1, -1,  0 },
    {  0,  0, -1 },
    {  0, -1, -1 },
    { -1,  0, -1 },
    { -1, -1, -1 },
  };

  for (int h=0; h<8; ++h)
  {
    int ii = i + idx[h][0], jj = j + idx[h][1], kk = k + idx[h][2];

    // printf("rank: %d  ii,jj,kk: %d %d %d\n", my_rank, ii, jj, kk);
    if (coor[kk][jj][ii].x < minc[0]) minc[0] = coor[kk][jj][ii].x;
    if (coor[kk][jj][ii].y < minc[1]) minc[1] = coor[kk][jj][ii].y;
    if (coor[kk][jj][ii].z < minc[2]) minc[2] = coor[kk][jj][ii].z;

    if (coor[kk][jj][ii].x > maxc[0]) maxc[0] = coor[kk][jj][ii].x;
    if (coor[kk][jj][ii].y > maxc[1]) maxc[1] = coor[kk][jj][ii].y;
    if (coor[kk][jj][ii].z > maxc[2]) maxc[2] = coor[kk][jj][ii].z;
  }
}


// Set the cells to each wall to user->buoyTempRef.
void cbuoySetFaceBC(UserCtx *user, PetscReal ***Conc)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
            lxs, lxe, lys, lye, lzs, lze);

  PetscInt i, j, k;
  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++)
  {
    if (k == 0 || j == 0 || i == 0 || k == mz-1 || j == my-1 || i == mx-1)
      Conc[k][j][i] = user->buoyTempRef;
  }
}

#include "buoyancy.test.BC.h"

// Sets the temperature boundary conditions which happen
// to be inside the mesh.
void cbuoyTestBC(UserCtx *user, PetscReal ***Conc)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  // static bool once = true;

  // Set the cells to each wall to user->buoyTempRef.
  cbuoySetFaceBC(user, Conc);

//   // Set temp in all cell to user->buoyTempRef except for the center z (j) plane
//   // that is set to user->buoyTemp
//   PetscInt i, j, k;
//   for (k=zs; k<ze; k++)
//   for (j=ys; j<ye; j++)
//   for (i=xs; i<xe; i++)
//   {
//     bool log = (i >= 20 && i <= 80 && j >= 20 && j <= 80 && k >= 10 && k <= 11);
//     printSequentialvaif(once && log, "k:  %4d  j:  %4d, i:  %4d\n", k, j, i);
//     if (log)
//       Conc[k][j][i] = user->buoyTemp;
//   }


  // Set temp in cells identified in buoyancy.test.BC.h.
  for (int imn=0; imn<buoyTestBCArrayCnt; ++imn)
  {
    int k = buoyTestBCArray[imn][0];
    int j = buoyTestBCArray[imn][1];
    for (int i=buoyTestBCArray[imn][2]; i<buoyTestBCArray[imn][3]; ++i)
    {
      bool log = (k >= zs && k < ze && j >= ys && j < ye && i >= xs && i < xe);
      if (log)
        Conc[k][j][i] = user->buoyTemp;
      // printSequentialvaif(once && log, "k:  %4d  j:  %4d, i:  %4d\n", k, j, i);
    }
  }

  // once = false;
}


#include "buoyancy.full.BC.h"


// Sets the temperature boundary conditions which happen
// to be inside the mesh.
void cbuoyMouthNoseBC(UserCtx *user, PetscReal ***Conc)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  // printSequentialva("zs: %4d  ze: %4d  ys: %4d  ye: %4d  xs: %4d  xe: %4d\n",
  //                   zs, ze, ys, ye, xs, xe);

  // Set the cells to each wall to user->buoyTempRef.
  cbuoySetFaceBC(user, Conc);

  // Set temp in cells identified in buoyancy.full.BC.h.
  for (int imn=0; imn<mouthNoseBCArrayCnt; ++imn)
  {
    int k = mouthNoseBCArray[imn][0];
    int j = mouthNoseBCArray[imn][1];
    for (int i=mouthNoseBCArray[imn][2]; i<mouthNoseBCArray[imn][3]; ++i)
    {
      bool log = (k >= zs && k < ze && j >= ys && j < ye && i >= xs && i < xe);
      if (log)
        Conc[k][j][i] = user->buoyTemp;
      // printSequentialvaif(log, "k:  %4d  j:  %4d, i:  %4d\n", k, j, i);
    }
  }

  // TECOutVecRank("Conc", "T", user, Conc);
}


void Conv_Diff_BC(UserCtx *user)
{
  DM            da = user->da, fda = user->fda;
  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions
  PetscInt      i, j, k, ibi;

  PetscInt      lxs, lxe, lys, lye, lzs, lze;

  //double nu = 1./user->ren;
  PetscReal     ***aj, ***rho, ***mu;
  PetscReal     ***lnu_t;
  extern PetscInt ti;
  //Cmpnts2 ***K_Omega;
  Vec Coor;
  Cmpnts        ***csi, ***eta, ***zet, ***ucat, ***coor;
  PetscReal     ***nvert, ***ustar;
  PetscReal     ***Conc;

  double nu_t, nu_tt;

  DMDAGetLocalInfo(da, &info);
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

  DMDAVecGetArray(fda, user->lCsi, &csi);
  DMDAVecGetArray(fda, user->lEta, &eta);
  DMDAVecGetArray(fda, user->lZet, &zet);
  DMDAVecGetArray(fda, user->lUcat, &ucat);

  if(rans || les) DMDAVecGetArray(da, user->lNu_t, &lnu_t);

  //DAGetGhostedCoordinates(da, &Coor);
  //DAVecGetArray(fda, Coor, &coor);
  DMDAGetCoordinates(da, &Coor);
  DMDAVecGetArray(fda, Coor, &coor);

  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(da, user->lAj, &aj);
  DMDAVecGetArray(da, user->lUstar, &ustar);

  DMDAVecGetArray(da, user->lConc, &Conc);

  if(levelset) {
    DMDAVecGetArray(da, user->lDensity, &rho);
    DMDAVecGetArray(da, user->lMu, &mu);
  }


  // The following code was added for my NYC runs. It sets the
  // concentration in a sphere near the NY Stock Exchange to 1.
#if 0

  // I turned it off on 2022-11-17 when I added the mouth/nose
  // buoyancy calculation.

  // Does the conc source information need to be initialized.
  // WRO 2020-04-20

#define TEST_LOC 0
  int my_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);


  if (!conc_src_init)
  {
    conc_src_init = true;

    Vec Cent = user->Cent;
    Cmpnts ***cent;
    DMDAVecGetArray(fda, user->Cent, &cent);

    // debug
    // printf("rank: %d  xs: %d  ys: %d  zs: %d  xe: %d  ye: %d  ze: %d\n",
    //        my_rank, xs, ys, zs, xe, ye, ze);
    // printf("rank: %d  lxs: %d  lys: %d  lzs: %d  lxe: %d  lye: %d  lze: %d\n",
    //        my_rank, lxs, lys, lzs, lxe, lye, lze);
    // MPI_Barrier(PETSC_COMM_WORLD);

    // If i j k are specified, see if it is on this processor.  If not,
    // set i value to -2 for eventual removal later.
    // printf("rank: %d checkpoint 1\n", my_rank);
    for (auto & h : conc_array)
    {
      if (h.type == 1)
      {
        if (h.i < lxs || h.i >= lxe ||
            h.j < lys || h.j >= lye ||
            h.k < lzs || h.k >= lze)
          h.dlete = true;
        else
        {
          // printf("rank: %d  i,j,k: %d %d %d is in this processor.\n", my_rank, h.i, h.j, h.k);
          // printf("rank: %d checkpoint 1a\n", my_rank);
          h.x = cent[h.k][h.j][h.i].x;
          h.y = cent[h.k][h.j][h.i].y;
          h.z = cent[h.k][h.j][h.i].z;
        }
      }
    }


    // If x y z are specified, first see if the location is inside
    // any cell on this processor.  This is done by finding a cartisian
    // bounding box around each cell.  If the location is inside the
    // bounding box, it's i j k are used.  This has two drawbacks:
    //   1. This may find the wrong cell if the cells are distorted
    //      or the mesh doesn't align with the cartisian coordinate
    //      system.
    //   2. We use the first one found which may not be the right one.
    // If you run into any of these problems, specify the i j k instead.
    // First see if there are any x y z entries.
    // printf("rank: %d checkpoint 2\n", my_rank);
    int entries = 0;
    // printf("rank: %d checkpoint tmp1 %d\n", my_rank, conc_array.size());
    for (auto & h : conc_array)
    {
      // printf("rank: %d checkpoint tmp2 %d\n", my_rank, &h - &*conc_array.begin());
      if (h.type == 2)
#if TEST_LOC
        ++entries;
#else
      h.dlete = true;
#endif
    }

    // printf("rank: %d checkpoint 2a\n", my_rank);

    if (entries > 0)
    {
      for (k=lzs; k<lze; k++)
      for (j=lys; j<lye; j++)
      for (i=lxs; i<lxe; i++)
      {
        // printf("rank: %d checkpoint2b %d %d %d  %f %f %f  %f %f %f\n",
        //        my_rank, i, j, k,
        //        coor[k][j][i].x, coor[k][j][i].y, coor[k][j][i].z,
        //        cent[k][j][i].x, cent[k][j][i].y, cent[k][j][i].z);
        double minc[3], maxc[3];
        coorMinMax(i, j, k, coor, minc, maxc);

        for (auto & h : conc_array)
        {
          if (h.x < minc[0] || h.x > maxc[0]  ||
              h.y < minc[1] || h.y > maxc[1]  ||
              h.z < minc[2] || h.z > maxc[2])
            break;
        }
      }
    }


    // printf("rank: %d checkpoint 3\n", my_rank);
    // Remove all entries with conc_array[].dlete true, they aren't on
    // this processor)
    for (int h=conc_array.size()-1; h>=0; --h)
    {
      if (conc_array[h].dlete)
        conc_array.erase(conc_array.begin() + h);
    }
    
//  PetscPrintf(PETSC_COMM_WORLD,
    printf("Rank %d, Number of concentration source entries: %d\n",
           my_rank, conc_array.size());

    // debug
    // print entries for this processor.
    // printf("rank: %d checkpoint 4\n", my_rank);
    // if (conc_array.size() > 0)
    //   printf("rank: %d  size: %d\n", my_rank, conc_array.size());
    // for (auto & h : conc_array)
    //   h.dump( my_rank, &h - &*conc_array.begin());

    DMDAVecRestoreArray(fda, user->Cent, &cent);
    MPI_Barrier(PETSC_COMM_WORLD);
  }

// #define CONC_APPLY_AS_SOURCE
#ifndef CONC_APPLY_AS_SOURCE
  if (conc_src_present)
  {
    static bool printed = false;
    if (!printed)
    {
      printed = true;
      PetscPrintf(PETSC_COMM_WORLD,
                  "Concentration applied using boundary conditions.\n");
    }
    for (auto & h : conc_array)
      Conc[h.k][h.j][h.i] = h.conc;
  }

  else
#endif
#endif
  {
    // BC for Concentration_pollution_sespended sediment
    for (k=lzs; k<lze; k++)
    for (j=lys; j<lye; j++)
    for (i=lxs; i<lxe; i++) {       // pressure node; cell center

      double ren = user->ren;

      // Set the conc BCs at the inlet.
      // from saved inflow file
      if (inletprofile==1000 && k==1)
      {
        Conc[k-1][j][i] = Inlet_Conc;
      }
      else if ( user->bctype[4]==5 && k==1 && nvert[k][j][i]<=1.2)
      {
        if (density_current == 1)
          Conc[k][j][i] = Conc[k+1][j][i];
        else if (density_current == 2)
        {
          if(coor[k][j][i].z < 6.6666666667)
            Conc[k][j][i]= 1.0;
          else
            Conc[k][j][i]= 0.0;
        }
        else if (density_current == 3)
          Conc[k-1][j][i]= 1.0;
        else
          Conc[k-1][j][i] = 0.0;
      }


      // The following code was added for my NYC runs. It sets the
      // concentration in a sphere near the NY Stock Exchange to 1.
#if 0

      // I turned it off on 2022-11-17 when I added the mouth/nose
      // buoyancy calculation.

      // Now apply concentration source boundary conditions.  WRO

      // For 3.00mm mesh size...
      // R01: -2.528290e+00, 4.854100e-01, -4.638216e-01 CCS
      //      i=530 j=22 k=321
      // R02: -2.657620e+00, 2.521100e-01, -4.638216e-01 CCS
      //      i=414 j=22 k=257
      // R03: -2.788950e+00, 1.606000e-02, -4.638216e-01 CCS
      //       i=296 j=22 k=191

      // For 6.75mm mesh size...
      // R01: -2.52829, 0.48541, -0.46382158m CCS
      //      minus     0.47291  -0.47632158
      //      plus      0.49791  -0.45132158
      //      i=236 j=10 k=143
      // R02: -2.65762, 0.25211, -0.46382158m CCS
      //      minus     0.23961  -0.47632158
      //      plus      0.26461  -0.45132158
      //      i=184 j=10 k=114
      // R03: -2.78895, 0.01606, -0.46382158m CCS
      //      minus     0.00356  -0.47632158
      //      plus      0.02856  -0.45132158
      //      i=132 j=10 k=85
      if ( !(inletprofile==1000 && k==1) )
      {
        if (!density_current)
        {
          // For 3.00mm mesh size.
          static int ks[3] = { 321, 257, 191 };
          // For 6.75mm mesh size.
          // static int ks[3] = { 143, 114, 85 };
          static double cs[3][3] = { { -2.528, 0.485, -0.464 },
                                     { -2.658, 0.252, -0.464 },
                                     { -2.789, 0.016, -0.464 } };

          int idx = -1;
          for (int ii=0; ii<sizeof(ks)/sizeof(int); ++ii)
          {
            if (k == ks[ii])
            {
              idx = ii;
              break;
            }
          }
          if (idx != -1 && nvert[k][j][i]<=1.2)
          {
            double yc  = (coor[k][j][i].y   + coor[k][j-1][i].y
                          + coor[k][j][i-1].y + coor[k][j-1][i-1].y)
              * 0.25;
            double zc  = (coor[k][j][i].z   + coor[k][j-1][i].z
                          + coor[k][j][i-1].z + coor[k][j-1][i-1].z)
              * 0.25;
            if (sqrt(sqr(yc-cs[idx][1]) + sqr(zc - cs[idx][2]))
                <= 0.0035)
              Conc[k-1][j][i] = Inlet_Conc;
          }
        }
      }
#endif


      // "slip"  same as "Symetrical B.C."
      if ( user->bctype[0] == 10 && i==1 ) Conc[k][j][i-1] = Conc[k][j][i];
      if ( user->bctype[1] == 10 && i==mx-2 ) Conc[k][j][i+1] = Conc[k][j][i];
      if ( user->bctype[2] == 10 && j==1 ) Conc[k][j-1][i] = Conc[k][j][i];
      if ( SuspendedParticles && user->bctype[3] == 10 && j==my-2 ){
        double delz = coor[k][j][i].z - coor[k][j-1][i].z;
        double nu  = 1./ren;
        if (rans || les) {nu_t = PetscMax(0.0 , 0.5*(lnu_t[k][j][i]+lnu_t[k][j-1][i])); nu_tt = nu + sigma_phi * nu_t;}
        else {nu_t = nu; nu_tt = nu;}

        Conc[k][j+1][i] = Conc[k][j][i]/(1. + delz * w_s / nu_tt);
      }
      if ( !SuspendedParticles && user->bctype[3] == 10 && j==my-2 ) Conc[k][j+1][i] = Conc[k][j][i];
      if ( user->bctype[4] == 10 && k==1 ) Conc[k-1][j][i] = Conc[k][j][i];
      if ( user->bctype[5] == 10 && k==mz-2 ) Conc[k+1][j][i] = Conc[k][j][i];

      // for density current in a closed reservoir
      if ( user->bctype[4] == 1 && k==1 ) Conc[k][j][i] = Conc[k+1][j][i];
      //if ( user->bctype[4] == 1 && k==1 ) Conc[k-1][j][i] = Conc[k][j][i];
      if ( user->bctype[5] == 1 && k==mz-2 ) Conc[k+1][j][i] = Conc[k][j][i];




      // outflow
      if ( user->bctype[5] == 4 && k==mz-2 ) {
        Conc[k+1][j][i] = Conc[k][j][i];
      }

      // couette
      if ( user->bctype[3] == 12 && j==my-2 ) {
        //double dist = distance[k][j][i];
        //K_Omega[k][j][i].y = wall_omega(ren, dist);
        //if(j==my-2) K_Omega[k][j+1][i].x = - K_Omega[k][j][i].x;
      }

      // wall
      if ( user->bctype[0] == 1 && i<=1 ) {
        //double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
        //double dist = distance[k][j][i];
        //dist = 0.5/aj[k][j][i]/area;
        //K_Omega[k][j][i].y = wall_omega(ren, dist);
        //if(i==1) K_Omega[k][j][i-1].x = - K_Omega[k][j][i].x + 1.e-5;
        //K_Omega[k][j][i-1].y = 2*10*wall_omega(dist) - K_Omega[k][j][i].y;
        if(i==1) Conc[k][j][i] = Conc[k][j][i+1];
        if(i==1) Conc[k][j][i-1] = Conc[k][j][i];
      }
      if ( user->bctype[1] == 1 && i>=mx-2 ) {
        //double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
        //double dist = distance[k][j][i];// 0.5/aj[k][j][i]/area;
        //dist = 0.5/aj[k][j][i]/area;
        //K_Omega[k][j][i].y = wall_omega(ren, dist);
        //if(i==mx-2) K_Omega[k][j][i+1].x = - K_Omega[k][j][i].x + 1.e-5;
        //K_Omega[k][j][i+1].y = 2*10*wall_omega(dist) - K_Omega[k][j][i].y;
        if(i==mx-2) Conc[k][j][i] =  Conc[k][j][i-1];
        if(i==mx-2) Conc[k][j][i+1] =  Conc[k][j][i];
      }
      if ( user->bctype[2] == 1 && j<=1 ) {
        //double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
        //double dist = distance[k][j][i];// 0.5/aj[k][j][i]/area;
        //dist = 0.5/aj[k][j][i]/area;
        //K_Omega[k][j][i].y = wall_omega(ren, dist);
        //if(j==1) K_Omega[k][j-1][i].x = - K_Omega[k][j][i].x + 1.e-5;
        //K_Omega[k][j-1][i].y = 2*10*wall_omega(dist) - K_Omega[k][j][i].y;
        if(j==1) Conc[k][j][i] =  Conc[k][j+1][i];
        if(j==1) Conc[k][j-1][i] =  Conc[k][j][i];
      }
      if ( user->bctype[3] == 1 && j>=my-2 ) {
        //double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
        //double dist = distance[k][j][i];// 0.5/aj[k][j][i]/area;
        //dist = 0.5/aj[k][j][i]/area;
        //K_Omega[k][j][i].y = wall_omega(ren, dist);
        //if(j==my-2) K_Omega[k][j+1][i].x = - K_Omega[k][j][i].x + 1.e-5;
        //K_Omega[k][j+1][i].y = 2*10*wall_omega(dist) - K_Omega[k][j][i].y;
        if(j==my-2) Conc[k][j][i] =  Conc[k][j-1][i];
        if(j==my-2) Conc[k][j+1][i] =  Conc[k][j][i];
      }

      // wall function
      if( nvert[k][j][i]<1.1 && ( ( (user->bctype[0]==-1 || user->bctype[0]==-2) && i==1) || ( (user->bctype[1]==-1 || user->bctype[1]==-2) &&  i==mx-2) ) && (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) {
        /*
          double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
          double sb, sc;
          Cmpnts Ua, Uc;
          Ua.x = Ua.y = Ua.z = 0;
          sb = 0.5/aj[k][j][i]/area;
          if(i==1) {
          sc = 2*sb + 0.5/aj[k][j][i+1]/area;
          Uc = ucat[k][j][i+1];
          }
          else {
          sc = 2*sb + 0.5/aj[k][j][i-1]/area;
          Uc = ucat[k][j][i-1];
          }
        */
        /*
          double ni[3], nj[3], nk[3];
          Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
          if(i==mx-2) ni[0]*=-1, ni[1]*=-1, ni[2]*=-1;

          wall_function (1./ren, sc, sb, Ua, Uc, &ucat[k][j][i], &ustar[k][j][i], ni[0], ni[1], ni[2]);
        */
        //double utau = ustar[k][j][i];

        //double Kc = utau*utau/sqrt(0.09);
        //double Ob = utau/sqrt(0.09)/(kappa*sb);

        //K_Omega[k][j][i].x = Kc;
        //K_Omega[k][j][i].y = Ob;
        if(i==1)Conc[k][j][i] = Conc[k][j][i+1];
        //if(i==1)Conc[k][j][i-1] = Conc[k][j][i];
        if(i==mx-2)Conc[k][j][i] = Conc[k][j][i-1];
        //if(i==mx-2)Conc[k][j][i+1] = Conc[k][j][i];
      }

      if( nvert[k][j][i]<1.1 && ( ( (user->bctype[2]==-1 || user->bctype[2]==-2) && j==1) || ( (user->bctype[3]==-1 || user->bctype[3]==-2) &&  j==my-2) ) && (i!=0 && i!=mx-1 && k!=0 && k!=mz-1)) {
        if(mobile_bed && !immersed){
          double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
          double sb, sc;
          Cmpnts Ua, Uc;
          Ua.x = Ua.y = Ua.z = 0;

          sb = 0.5/aj[k][j][i]/area;

          if(j==1) {
            sc = 2*sb + 0.5/aj[k][j+1][i]/area;
            Uc = ucat[k][j+1][i];
          }
          else {
            sc = 2*sb + 0.5/aj[k][j-1][i]/area;
            Uc = ucat[k][j-1][i];
          }

          double ni[3], nj[3], nk[3];
          Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
          double nfx = nj[0], nfy = nj[1], nfz = nj[2];
          if(j==my-2) nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;

          wall_function (1./ren, sc, sb, Ua, Uc, &ucat[k][j][i], &ustar[k][j][i], nj[0], nj[1], nj[2]);

          double utau = ustar[k][j][i];
          //Cmpnts uc;
          //if(j==1) {uc = ucat[k][j+1][i];}
          //else {uc = ucat[k][j][i];}

          // Ali bed B.C.
          PetscInt VanRijn=0, Fredsoe=1, non_equilibrium = 1;
          //double U_bulk_vel = 1.0;//1.87;//0.593;//1.5;
          double U_bulk_vel = U_Bulk;//1.87;//0.593;//1.5;
          double ho = 0.152;//8.;//.3;
          double fluid_density = 1000.0;
          //double D50 = 0.0005;//0.0007;
          double D50 = d50;//0.0005;//0.0007;
          //double Deltab = 10.*D50;//0.08;//2.*D50;
          double Deltab = deltab;//10.*D50;//0.08;//2.*D50;
          double particle_sg = sed_density/fluid_density;
          double gi = 9.81;
          double visk = 1.e-6;
          double Angle_repose = 45. * 3.1415 / 180.;
          double CShields_param, Ctau_0, Custar_0, tau_0, tt, cb, bita, tmp;
          double betta, alfa, gamma, nu_tt;
          double sgd_tmp   = ( particle_sg - 1. ) * gi * D50;
          double Dstar = D50 * pow(((particle_sg-1.) * gi / ( visk * visk )), 0.3333333333 );

          int shields = 0, soulsby = 1; // soulseby and Whitehouse methods: see J Geoph. Res. vol 115, 2010, Chou & Fringer

          if(shields){

            if( Dstar <= 4. ) CShields_param = 0.24 / Dstar;
            if( Dstar > 4. && Dstar <= 10. ) CShields_param = 0.14 / pow( Dstar, 0.64 );
            if( Dstar > 10. && Dstar <= 20. ) CShields_param = 0.04 / pow( Dstar, 0.1 );
            if( Dstar > 20. && Dstar <= 150. ) CShields_param = 0.013 * pow( Dstar, 0.29 );
            if( Dstar > 150. ) CShields_param = 0.055;
          }

          if (soulsby) { CShields_param = 0.3/ ( 1. + 1.2 * Dstar) + 0.055 * (1. - exp (-0.02 * Dstar)) ; }

          //double tita_x = atan ( - nfx / nfz );
          //double tita_y = atan ( - nfy / nfz );
          //double tita1  = uc.x * sin (tita_x) + uc.y * sin (tita_y);
          //double tita2  = sqrt (uc.x * uc.x + uc.y * uc.y);
          //double tita3  = tita1 / tita2;
          //if (fabs(tita3) >= 1. || fabs(tita3)<= -1.) { tmp =1.;} else {
          //double tita   = asin (tita3);
          //       tmp    = sin ( tita + Angle_repose ) / sin (Angle_repose);}

          //CShields_param = CShields_param * tmp;



          if(VanRijn) {
            Ctau_0 = CShields_param * sgd_tmp * fluid_density;
            tau_0 =  fluid_density*(utau*U_bulk_vel)*(utau*U_bulk_vel);
            tt = (tau_0 - Ctau_0)/Ctau_0;
            tt = PetscMax( tt, 0. );
            cb =  0.015 *(D50/Deltab)*pow( tt, 1.5 )/pow( Dstar,0.3);
          }

          if(Fredsoe) {
            Ctau_0 = CShields_param;
            tau_0 = (utau*U_bulk_vel)*(utau*U_bulk_vel); // the inlet vel. increase by 1.5 times
            tau_0 = tau_0/(sgd_tmp);
            tt = tau_0 - Ctau_0;
            tt = PetscMax( tt, 1.e-7 );
            bita = PI * 0.51 / 6.;
            tt= tt*tt*tt*tt;
            bita=bita*bita*bita*bita;
            cb = pow((1.+ bita/tt),-0.25)*PI/6.;
            cb = PetscMax (0.0, cb);
          }
          if(non_equilibrium){
            double delz = coor[k][j+1][i].z - coor[k][j][i].z;
            double nu  = 1./ren;
            if(rans || les) {nu_t = PetscMax(0.0 , 0.5*(lnu_t[k][j][i]+lnu_t[k][j-1][i])); nu_tt = nu + sigma_phi * nu_t;}
            else {nu_t = nu; nu_tt = nu;}

            alfa = 1. - exp ( PetscMax ( 0.0, w_s * ( delz - Deltab / ho ) / nu_tt ) );
            alfa = PetscMax (0.0, alfa);
            betta = (1. - alfa) * ( w_s * delz / nu_tt);
            gamma = 1. - w_s * delz / nu_tt;
          }
          //PetscPrintf(PETSC_COMM_WORLD, "tmp_j=1   cb %e %e \n", tmp,cb );
          //PetscPrintf(PETSC_COMM_WORLD, "u*, tau_0, Ctau_0, Cb %e %e %e %e\n", utau, tau_0, Ctau_0, cb );
          //PetscPrintf(PETSC_COMM_WORLD, "Cb alfa betta gamma %e %e %e %e \n", cb, alfa, betta, gamma);
          if(j==1 && !non_equilibrium) Conc[k][j][i] = cb;
          if(j==1 && non_equilibrium) Conc[k][j][i] = Conc[k][j+1][i] + fabs(betta/gamma) * cb;

        } else {

          if(j==1)Conc[k][j][i] = Conc[k][j+1][i];
          //if(j==1)Conc[k][j-1][i] = Conc[k][j][i];
          if(j==my-2)Conc[k][j][i] = Conc[k][j-1][i];
          //if(j==my-2)Conc[k][j+1][i] = Conc[k][j][i];
        }
      }

      if ( nvert[k][j][i] > 1.1 ) Conc[k][j][i]= 0.;
    }
  }

  DMDAVecRestoreArray(da, user->lConc, &Conc);

  DMDALocalToLocalBegin(user->da, user->lConc, INSERT_VALUES, user->lConc);
  DMDALocalToLocalEnd(user->da, user->lConc, INSERT_VALUES, user->lConc);

  DMDAVecGetArray(user->da, user->lConc, &Conc);

  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++) {
    int flag=0, a=i, b=j, c=k;

    if(i_periodic && i==0) a=mx-2, flag=1;
    else if(i_periodic && i==mx-1) a=1, flag=1;

    if(j_periodic && j==0) b=my-2, flag=1;
    else if(j_periodic && j==my-1) b=1, flag=1;

    if(k_periodic && k==0) c=mz-2, flag=1;
    else if(k_periodic && k==mz-1) c=1, flag=1;


    if(ii_periodic && i==0) a=-2, flag=1;
    else if(ii_periodic && i==mx-1) a=mx+1, flag=1;

    if(jj_periodic && j==0) b=-2, flag=1;
    else if(jj_periodic && j==my-1) b=my+1, flag=1;

    if(kk_periodic && k==0) c=-2, flag=1;
    else if(kk_periodic && k==mz-1) c=mz+1, flag=1;

    if(flag) Conc[k][j][i] = Conc[c][b][a];
  }

  if(immersed){
    for(ibi=0; ibi<NumberOfBodies; ibi++)
    {
      extern IBMNodes *ibm_ptr;
      IBMNodes *ibm = &ibm_ptr[ibi];

      IBMListNode *current;
      current = user->ibmlist[ibi].head;
      while (current) {
        IBMInfo *ibminfo = &current->ibm_intp;
        int ni = ibminfo->cell;
        current = current->next;
        double sb = ibminfo->d_s, sc = sb + ibminfo->d_i;
        int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
        int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
        int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
        i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
        double sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
        //double Kc = (K_Omega[kp1][jp1][ip1].x * sk1 + K_Omega[kp2][jp2][ip2].x * sk2 + K_Omega[kp3][jp3][ip3].x * sk3);
        double C = (Conc[kp1][jp1][ip1] * sk1 + Conc[kp2][jp2][ip2] * sk2 + Conc[kp3][jp3][ip3] * sk3);
        double ren = user->ren;
        if(levelset) ren = rho[k][j][i]/ mu[k][j][i];

        double nfx = ibm->nf_x[ni];
        double nfy = ibm->nf_y[ni];
        double nfz = ibm->nf_z[ni];
        Cmpnts uc = ucat[k][j][i];

        if(wallfunction && ti>0) {
          //double utau = ustar[k][j][i];
          // Ali bed B.C.

          //int rigid = ibm->Rigidity[ni];
          int mobile = ibm->Mobility[ni];
          PetscInt Fringer = 0, MikiBC = 0, _MikiBC = 0;
          //if(mobile_bed && !rigid)
          //if(mobile_bed && mobile)
          //if(mobile_bed && ibm->Rigidity[ni]==0)
          if(mobile_bed && (ibm->nf_z[ni]> 0.99 || (sediment && ibm->Rigidity[ni]==0)))
          {
            double utau = ustar[k][j][i];

            PetscInt VanRijn=0, Fredsoe=1, non_equilibrium = 0;

            //if(LiveBed) non_equilibrium = 1;

            //double U_bulk_vel = 1.0;//1.87;//1.1;// Max Value0.593; //Ubulk=0.501m/s;
            double U_bulk_vel = U_Bulk;//1.0;//1.87;//1.1;// Max Value0.593; //Ubulk=0.501m/s;
            double ho = 0.152;//8.0;//0.152;
            double fluid_density = 1000.0;
            //double D50 = 0.0005; //0.00025;
            double D50 = d50;//0.0005; //0.00025;
            //double Deltab = 10.*D50;//2.*D50;
            double Deltab = deltab;//10.*D50;//2.*D50;
            double particle_sg = sed_density/fluid_density;
            double porosity = 0.45;
            double gi = 9.81;
            double visk = 1.e-6;
            double Angle_repose = 45. * 3.1415 / 180.;
            double CShields_param, Ctau_0, Custar_0, tau_0, tt, cb, bita, tmp, coeff;
            double betta, alfa, gamma;
            double sgd_tmp   = ( particle_sg - 1. ) * gi * D50;
            double Dstar = D50 * pow(((particle_sg-1.) * gi / ( visk * visk )), 0.3333333333 );
            int shields = 0, soulsby = 1, old_method = 0, whitehouse = 1; // soulseby and Whitehouse methods: see J Geoph. Res. vol 115, 2010, Chou & Fringer


            if(shields){
              if( Dstar <= 4. ) CShields_param = 0.24 / Dstar;
              if( Dstar > 4. && Dstar <= 10. ) CShields_param = 0.14 / pow( Dstar, 0.64 );
              if( Dstar > 10. && Dstar <= 20. ) CShields_param = 0.04 / pow( Dstar, 0.1 );
              if( Dstar > 20. && Dstar <= 150. ) CShields_param = 0.013 * pow( Dstar, 0.29 );
              if( Dstar > 150. ) CShields_param = 0.055;
            }
            if (soulsby) { CShields_param = 0.3/ ( 1. + 1.2 * Dstar) + 0.055 * (1. - exp (-0.02 * Dstar)) ; }

            /*
              double tita_x = atan ( - nfx / nfz );
              double tita_y = atan ( - nfy / nfz );
              double tita1  = uc.x * sin (tita_x) + uc.y * sin (tita_y);
              double tita2  = sqrt (uc.x * uc.x + uc.y * uc.y);
              double tita3  = tita1 / tita2;
              if (fabs(tita3) >= 1. || fabs(tita3)<= -1.) { tmp =1.;} else {
              double tita   = asin (tita3);
              tmp    = sin ( tita + Angle_repose ) / sin (Angle_repose);}
              CShields_param *= tmp;
            */

            if(!Fringer){

              if(VanRijn) {
                Ctau_0 = CShields_param * sgd_tmp * fluid_density;
                tau_0 =  fluid_density*(utau*U_bulk_vel)*(utau*U_bulk_vel);
                tt = (tau_0 - Ctau_0)/Ctau_0;
                tt = PetscMax( tt, 0. );
                cb =  0.015 *(D50/Deltab)*pow( tt, 1.5 )/pow( Dstar,0.3);
              }

              if(Fredsoe) {
                Ctau_0 = CShields_param;
                tau_0 = (utau*U_bulk_vel)*(utau*U_bulk_vel); // the inlet vel. increase by 1.5 times
                tau_0 = tau_0/(sgd_tmp);
                tt = tau_0 - Ctau_0;
                tt = PetscMax( tt, 1.e-7 );
                bita = PI * 0.51 / 6.;
                tt= tt*tt*tt*tt;
                bita=bita*bita*bita*bita;
                cb = pow((1.+ bita/tt),-0.25)*PI/6.;
                cb = PetscMax (0.0, cb);
              }

              if(non_equilibrium){
                //double delz = coor[k][j+1][i].z - coor[k][j][i].z;
                double delz = sc - sb;
                double nu  = 1./ren;
                if(rans || les) {nu_t = PetscMax(0.0 , 0.5*(lnu_t[k][j][i]+lnu_t[k][j-1][i])); nu_tt = nu + sigma_phi * nu_t;}
                else {nu_t = nu; nu_tt = nu;}
                /*alfa = 1. - exp ( w_s * PetscMax( 0., delz - ( Deltab / ho ) ) / nu_tt);
                  alfa = PetscMax ( 0.0, alfa);
                  alfa = PetscMin ( 1.0, alfa);
                  betta = (1. - alfa) * ( w_s * delz / nu_tt);
                  gamma = 1. - w_s * delz / nu_tt;
                  coeff = PetscMin (fabs(betta/gamma), 1.0); */
                coeff = PetscMin (fabs(w_s * delz / nu_tt), 1.0);
              }
              if(!non_equilibrium) Conc[k][j][i] = cb;
              /* if(non_equilibrium){
                 if(C > cb) Conc[k][j][i] = C - coeff * cb;
                 if(C <= cb) Conc[k][j][i] = C + coeff * cb;
                 }*/
              if(non_equilibrium) Conc[k][j][i] = C + coeff * cb;
              //if(non_equilibrium) Conc[k][j][i] = Conc[k][j+1][i] + (betta/gamma) * cb;

              //const double yplus_min = 0.25;
              //sb = PetscMax( sb, 1./ren * yplus_min / utau ); // prevent the case sb=0
              //double K, Ob;//, Om;
              // Wilcox pp.108-109
              //K = utau*utau/sqrt(0.09);
              //Ob = utau/sqrt(0.09)/(kappa*sb);
              //K_Omega[k][j][i].x = K;
              //K_Omega[k][j][i].y = Ob;

            } // if not-Fringer

            if(Fringer)    { //This will consider the Pick up function and other stuff entraining sediment into the flow.
              //to account for the concentration change effect on the mommentom---density current

              Ctau_0 = CShields_param * sgd_tmp * fluid_density;
              tau_0 =  fluid_density*(utau*U_bulk_vel)*(utau*U_bulk_vel);
              tt = (tau_0 - Ctau_0)/Ctau_0;
              tt = PetscMax( tt, 0. );
              double P_k = 0.00033 * pow( tt, 1.5 ) * pow( Dstar,0.3) * sqrt(sgd_tmp);

              double C_extr;
              if(C >= Conc[k][j][i]) {C_extr = PetscMax(C * (sc/(sc-sb)) - Conc[k][j][i] * (sb/(sc-sb)), 0.0);}
              else {C_extr = PetscMax(Conc[k][j][i] * (sc/(sc-sb)) - C * (sb/(sc-sb)), 0.0);}

              double delz = coor[k][j+1][i].z - coor[k][j][i].z;

              double nu  = 1./ren;
              if(rans || les) {
                nu_t = PetscMax(0.0 , 0.5*(lnu_t[k][j][i]+lnu_t[k][j+1][i]));
                nu_tt = nu + sigma_phi * nu_t;
              } else {
                nu_t = nu;
                nu_tt = nu;
              }
              double wbar = 0.5 * (ucat[k][j][i].z + ucat[k][j+1][i].z);
              //double wbar = 0.0;
              double cbar = 0.5 * (Conc[k][j][i] + Conc[k][j+1][i]);
              double LeoTerm = wbar * cbar - nu_tt * (Conc[k][j+1][i] - Conc[k][j][i])/delz - w_s * cbar;

              Conc[k][j][i] = Conc[k][j][i] - user->dt * LeoTerm / delz + user->dt * (P_k - w_s * C_extr) / delz;

              //Streeter Boundary Condition: (1/nu_t).partial(C)/Partial(z) = -P_k
              //Conc[k][j][i] = C + P_k * (sc-sb) / nu_tt;

              if (Conc[k][j][i] > porosity) Conc[k][j][i] = porosity;
              if (Conc[k][j][i] < 0.0) Conc[k][j][i] = 0.0;
              if (Conc[k][j][i] < C) Conc[k][j][i] = C;
            }

          }  //if mobile-bed

          else if(mobile_bed && ibm->Rigidity[ni]) {Conc[k][j][i] = C;}

          else if(!mobile_bed && zero_grad){Conc[k][j][i] = C;}


          else if(!mobile_bed && !zero_grad && MikiBC){

            // Miki_Hondzo's B.C. just on sediments
            if(ibm->nf_z[ni]<0.8 || ((ibm->cent_x[ni]<45. && ibm->cent_z[ni]>6.98) || (ibm->cent_y[ni]<55.5 && ibm->cent_x[ni]>75.2 && ibm->cent_z[ni]>6.62)))
            {
              Conc[k][j][i] = C;
            } else {

              double utau = ustar[k][j][i];
              double U_bulk_vel = 0.16; //0.032;
              double ho = 0.16;
              double Re_tau = (utau * U_bulk_vel) * ho * 1000000.;

              double c_v = Depth_Averaged_Conc(Conc, nvert, i, j, k, lye);

              double c_s = c_v / ( 1.0 + 0.67 * exp ((Re_tau - 17300.) * (Re_tau - 17300.)/(-2.1 * 100000000.)));

              //PetscPrintf(PETSC_COMM_WORLD, "c_v, c_s Re %e %e %e\n", c_v, c_s, Re_tau);

              Conc[k][j][i] = c_s;}

          }
          else if(!mobile_bed && !zero_grad && _MikiBC){

            // Miki_Hondzo's Newly developed B.C. for NO3 (flux based BC) on the sediment bed material
            if(ibm->nf_z[ni]<0.9 || ibi)
            {
              Conc[k][j][i] = C;
            } else {

              double utau = ustar[k][j][i];
              double U_bulk_vel = 0.16667;
              double ho = (coor[k][lye][i].z - coor[k][j][i].z) * 0.16;

              if(ho < 0.0) PetscPrintf(PETSC_COMM_WORLD, "\n error ****** Flow depth cannot be  negative **** error\n");

              double Re_tau = (utau * U_bulk_vel) * ho * 1000000.;

              double c_v = Depth_Averaged_Conc(Conc, nvert, i, j, k, lye);

              //PetscPrintf(PETSC_COMM_WORLD, "c_v, c_s Re %e %e %e\n", c_v, c_s, Re_tau);
              double delz = coor[k][j+1][i].z - coor[k][j][i].z;
              double nu  = 1./ren;
              if(rans || les) {
                nu_t = PetscMax(0.0 , 0.5*(lnu_t[k][j][i]+lnu_t[k][j+1][i]));
                nu_tt = nu + sigma_phi * nu_t;
              } else {
                nu_t = nu;
                nu_tt = nu;
              }
              double rho_b = 1.87; //=1.63 +- 0.49 g/cm3
              double BOM = 0.162;//=0.102 +- 0.12 g/cm2

              double u_star = utau * U_bulk_vel;

              if(ho = 0.0) PetscPrintf(PETSC_COMM_WORLD, "\n error ****** Flow depth is zero **** error\n");
              if(c_v = 0.0) PetscPrintf(PETSC_COMM_WORLD, "\n error ****** Averaged No3 concentration is zero **** error\n");

              //double Flux_No3 = 1.23 * pow(10.,-6.) * (u_star * c_v) * pow((u_star * ho * 1000000.),-(9./11.)) * pow((BOM/(ho*c_v)),(3./5.)) * pow((rho_b/c_v),(5./7.)) + 0.024; former simulation done with this
              //double Flux_No3 = 1.23 * pow(10.,-6.) * (u_star * c_v) * pow((u_star * ho * 1000000.),-(9./11.)) * pow((BOM/(ho*c_v)),(3./5.)) * pow((rho_b/c_v),(5./7.)) - 0.024;
              double Flux_No3 = 1.0 * pow(10.,-9.6) * pow((u_star * ho * 1000000.),-(9./11.)) * pow((10000.*BOM/(ho*c_v)),(3./5.)) * pow((1000000.*rho_b/c_v),(5./7.));

              if (nu_tt > 0.) {Conc[k][j][i] = PetscMax (0.0, (C-fabs(Flux_No3 * (sc - sb) / nu_tt)));
                if(Conc[k][j][i] > C) Conc[k][j][i] = C;
              }
              if (nu_tt <= 0.) Conc[k][j][i] = C;
            }

          }
          else {Conc[k][j][i] = C;}

        }  //WallFunction

        if(!wallfunction) {
          //const double yplus_min = 0.25;
          //double utau = ustar[k][j][i];
          //K_Omega[k][j][i].x = Kc * sb / sc;
          //sb = PetscMax( sb, 1./ren * yplus_min / utau );     // prevent the case sb=0
          //K_Omega[k][j][i].y = wall_omega(ren, sb);

          //if ( K_Omega[k][j][i].x < 0 ) K_Omega[k][j][i].x = utau*utau/sqrt(0.09);
          Conc[k][j][i] = C;
        }
        if(user->bctype[4]==5 && k==1) Conc[k][j][i] = Conc[k-1][j][i];
      };
    }
  }

  // Apply mouth/nose temperature boundary condition.
  if (user->ctemp)
  {
    if (user->buoyTest)
      cbuoyTestBC(user, Conc);

    if (user->buoyMouthNose)
      cbuoyMouthNoseBC(user, Conc);
  }


  DMDAVecRestoreArray(fda, user->lCsi, &csi);
  DMDAVecRestoreArray(fda, user->lEta, &eta);
  DMDAVecRestoreArray(fda, user->lZet, &zet);
  DMDAVecRestoreArray(fda, user->lUcat, &ucat);

  if(rans || les) DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);

  DMDAVecRestoreArray(fda, Coor, &coor);

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(da, user->lAj, &aj);
  DMDAVecRestoreArray(da, user->lUstar, &ustar);

  DMDAVecRestoreArray(user->da, user->lConc, &Conc);

  if(levelset) {
    DMDAVecRestoreArray(da, user->lDensity, &rho);
    DMDAVecRestoreArray(da, user->lMu, &mu);
  }
  DMDALocalToLocalBegin(user->da, user->lConc, INSERT_VALUES, user->lConc);
  DMDALocalToLocalEnd(user->da, user->lConc, INSERT_VALUES, user->lConc);
};

PetscErrorCode FormFunction_Conv_Diff(SNES snes, Vec Conc, Vec Rhs, void *ptr)
{
  UserCtx *user = (UserCtx*)ptr;

  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions
  PetscInt      i, j, k;
  PetscInt      lxs, lxe, lys, lye, lzs, lze;

  PetscReal     ***nvert;
  //Cmpnts2 ***rhs;
  PetscReal  ***rhs;

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


  DMGlobalToLocalBegin(user->da, Conc, INSERT_VALUES, user->lConc);
  DMGlobalToLocalEnd(user->da, Conc, INSERT_VALUES, user->lConc);

  Conv_Diff_BC(user);

  if(SuspendedParticles)SettlingVelocityCorrection(user);

  RHS_Conv_Diff(user, Rhs);


  VecAXPY(Rhs, -1/user->dt, Conc);
  VecAXPY(Rhs, 1/user->dt, user->Conc_o);

  DMDAVecGetArray(user->da, user->lNvert, &nvert);
  DMDAVecGetArray(user->da, Rhs, &rhs);

  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++) {
    if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 || nvert[k][j][i]>0.1) { //it was 0.1 for k-omega
      rhs[k][j][i] = 0.;
    }

    // couette
    //if ( user->bctype[3] == 12 && j==my-2 ) rhs[k][j][i].y = 0;

    //wall_omega
    /*
      if( i<=1 && user->bctype[0]==1 ) rhs[k][j][i].y = 0;
      if( i>=mx-2 && user->bctype[1]==1 ) rhs[k][j][i].y = 0;

      if( j<=1 && user->bctype[2]==1 ) rhs[k][j][i].y = 0;
      if( j>=my-2 && user->bctype[3]==1 ) rhs[k][j][i].y = 0;

      if( k==1 && user->bctype[4]==1 ) rhs[k][j][i].y = 0;
      if( k==mz-2 && user->bctype[5]==1 ) rhs[k][j][i].y = 0;
    */

    if( i>=mx-2 && user->bctype[1]==1 ) rhs[k][j][i] = 0.;
    if( j<=1 && user->bctype[2]==1 ) rhs[k][j][i] = 0.;

    if( j<=1 && user->bctype[2]==1 ) rhs[k][j][i] = 0.;
    if( j>=my-2 && user->bctype[3]==1 ) rhs[k][j][i] = 0.;

    if( k==1 && user->bctype[4]==1 ) rhs[k][j][i] = 0.;
    if( k==mz-2 && user->bctype[5]==1 ) rhs[k][j][i] = 0.;

    // wall function k, omega
    if (    ( i==1 && user->bctype[0] == -1 ) || ( i==mx-2 && user->bctype[1] == -1 ) ||
            ( j==1 && user->bctype[2] == -1 ) || ( j==my-2 && user->bctype[3] == -1 ) ||
            ( k==1 && user->bctype[4] == -1 ) || ( k==mz-2 && user->bctype[5] == -1 ) ||
            ( i==1 && user->bctype[0] == -2 ) || ( i==mx-2 && user->bctype[1] == -2 ) ||
            ( j==1 && user->bctype[2] == -2 ) || ( j==my-2 && user->bctype[3] == -2 ) ||
            ( k==1 && user->bctype[4] == -2 ) || ( k==mz-2 && user->bctype[5] == -2 )
      ) {
      //rhs[k][j][i].x = 0;    if you compute this value using an equation for the i==1 then the Rhs of it should be set to 0
      //rhs[k][j][i].y = 0;
      rhs[k][j][i] = 0.;
    }
  }

#ifdef CONC_APPLY_AS_SOURCE
  // WRO
  if (conc_src_present)
  {
    static bool printed = false;
    if (!printed)
    {
      printed = true;
      PetscPrintf(PETSC_COMM_WORLD,
                  "Concentration applied using source terms.\n");
    }
    for (auto & h : conc_array)
      rhs[h.k][h.j][h.i] += h.conc / user->dt;
  }
  // END WRO
#endif

  DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, Rhs, &rhs);

  return(0);
}

int snes_convdiff_created=0;
Vec r_convdiff;
Mat J_convdiff;
SNES snes_convdiff;

void Solve_Conv_Diff(UserCtx *user, double & norm)
{

  KSP ksp;
  PC pc;

  int bi=0;
  double tol=1.e-6;//5.e-5;//1.e-6

  if(!snes_convdiff_created) {
    snes_convdiff_created=1;

    VecDuplicate(user[bi].Conc, &r_convdiff);
    SNESCreate(PETSC_COMM_WORLD,&snes_convdiff);
    SNESSetFunction(snes_convdiff,r_convdiff,FormFunction_Conv_Diff,(void *)&user[bi]);
    MatCreateSNESMF(snes_convdiff, &J_convdiff);
    SNESSetJacobian(snes_convdiff,J_convdiff,J_convdiff,MatMFFDComputeJacobian,(void *)&user[bi]);
    SNESSetType(snes_convdiff, SNESTR);                 //SNESTR,SNESLS
    SNESSetMaxLinearSolveFailures(snes_convdiff,10000);
    SNESSetMaxNonlinearStepFailures(snes_convdiff,10000);
    SNESKSPSetUseEW(snes_convdiff, PETSC_TRUE);
    SNESKSPSetParametersEW(snes_convdiff,3,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    SNESSetTolerances(snes_convdiff,PETSC_DEFAULT,tol,PETSC_DEFAULT,50,50000);

    SNESGetKSP(snes_convdiff, &ksp);
    KSPSetType(ksp, KSPGMRES);

    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCNONE);

    int maxits=50/*20 or 10000*/;       double rtol=tol, atol=PETSC_DEFAULT, dtol=PETSC_DEFAULT;
    KSPSetTolerances(ksp,rtol,atol,dtol,maxits);
  }

  extern PetscErrorCode MySNESMonitor(SNES snes,PetscInt n,PetscReal rnorm,void *dummy);
  SNESMonitorSet(snes_convdiff,MySNESMonitor,PETSC_NULL,PETSC_NULL);

  PetscPrintf(PETSC_COMM_WORLD, "\nSolving Convection-Diffusion ...\n");

  SNESSolve(snes_convdiff, PETSC_NULL, user[bi].Conc);

  SNESGetFunctionNorm(snes_convdiff, &norm);
  PetscPrintf(PETSC_COMM_WORLD, "\nConcentration SNES residual norm=%.5e\n\n", norm);

  DMDALocalInfo info;
  PetscInt      xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt      mx, my, mz; // Dimensions in three directions
  PetscInt      i, j, k;
  PetscInt      lxs, lxe, lys, lye, lzs, lze;

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

  DMGlobalToLocalBegin(user->da, user->Conc, INSERT_VALUES, user->lConc);
  DMGlobalToLocalEnd(user->da, user->Conc, INSERT_VALUES, user->lConc);

  PetscReal ***conc, ***lconc;

  DMDAVecGetArray(user->da, user->Conc, &conc);
  DMDAVecGetArray(user->da, user->lConc, &lconc);

  if(periodic)
    for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
    for (i=xs; i<xe; i++) {
      int flag=0, a=i, b=j, c=k;

      if(i_periodic && i==0) a=mx-2, flag=1;
      else if(i_periodic && i==mx-1) a=1, flag=1;

      if(j_periodic && j==0) b=my-2, flag=1;
      else if(j_periodic && j==my-1) b=1, flag=1;

      if(k_periodic && k==0) c=mz-2, flag=1;
      else if(k_periodic && k==mz-1) c=1, flag=1;


      if(ii_periodic && i==0) a=-2, flag=1;
      else if(ii_periodic && i==mx-1) a=mx+1, flag=1;

      if(jj_periodic && j==0) b=-2, flag=1;
      else if(jj_periodic && j==my-1) b=my+1, flag=1;

      if(kk_periodic && k==0) c=-2, flag=1;
      else if(kk_periodic && k==mz-1) c=mz+1, flag=1;


      if(flag) conc[k][j][i] = lconc[c][b][a];
    }
  DMDAVecRestoreArray(user->da, user->Conc, &conc);
  DMDAVecRestoreArray(user->da, user->lConc, &lconc);

  DMGlobalToLocalBegin(user->da, user->Conc, INSERT_VALUES, user->lConc);
  DMGlobalToLocalEnd(user->da, user->Conc, INSERT_VALUES, user->lConc);

  Conv_Diff_BC(user);

  if(SuspendedParticles) SettlingVelocityCorrectionBack(user);

  DMLocalToGlobalBegin(user->da, user->lConc, INSERT_VALUES, user->Conc);
  DMLocalToGlobalEnd(user->da, user->lConc, INSERT_VALUES, user->Conc);

};
