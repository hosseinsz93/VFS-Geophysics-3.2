/*
  This temperature calculation has some errors in it and I had to
  give up using it.  Use the bouyancy options found in convection_diffusion.c
 */
#include "variables.h"
#include "MeshParms.h"
#include "TECOutput.h"

#include <math.h>
#include <memory>
#include <iostream>

#include "PrintSequential.h"

using namespace std;
extern PetscInt immersed, NumberOfBodies, ti, tistart, wallfunction;
extern PetscErrorCode MyKSPMonitor1(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy);
extern PetscInt inletprofile;

//double solid=0.1;


// Support routines for buoyancy calculation in its own namespace.

class BuoySupport
{
  public:
    struct bc
    {
      int x[4];

      int       & operator[](int idx)       { return x[idx]; }
      int const & operator[](int idx) const { return x[idx]; }
    };

    typedef std::unique_ptr <BuoySupport> uni;
    typedef std::vector <bc> vec;

    static BuoySupport * get()
    {
      if (!bs)
        bs = new BuoySupport; 

      return bs;
    }

    void readFile(UserCtx * user)
    {
      if (fileRead)
        return;

      std::ifstream fi(user->buoyFilename);
      if (!fi.good())
      {
        PetscPrintf(PETSC_COMM_WORLD, "Couldn't open file %s.\n", user->buoyFilename.c_str());
        exit(1);
      }

      int cnt, vsize;
      int lcnt = -1;
      std::string buff;

      while(!fi.eof())
      {
        ++lcnt;
        std::getline(fi, buff);

        // The first line contains the number of entries in the file.
        if (lcnt == 0)
        {
          cnt = sscanf(buff.c_str(), "%d", &vsize);
          if (cnt < 1)
          {
            PetscPrintf(PETSC_COMM_WORLD, "Error reading file %s, line %d.\n", user->buoyFilename.c_str(), lcnt+1);
            exit(1);
          }
          bcs.reserve(vsize);
          continue;
        }

        BuoySupport::bc ent;
        cnt = sscanf(buff.c_str(), "%d, %d, %d, %d", &ent[0], &ent[1], &ent[2], &ent[3]);
        if (cnt < 4)
        {
          PetscPrintf(PETSC_COMM_WORLD, "Error reading file %s, line %d.\n", user->buoyFilename.c_str(), lcnt+1);
          exit(1);
        }
        bcs.push_back(ent);

        // If this is the last entry, don't read the file anymore.
        if (lcnt == vsize)
          break;
      }

      if (lcnt < vsize)
      {
        PetscPrintf(PETSC_COMM_WORLD, "Error reading file %s, line count if not correct.\n", user->buoyFilename.c_str());
        exit(1);
      }

      fileRead = true;
    }

    vec::const_iterator begin() const { return bcs.begin(); }
    vec::const_iterator end() const   { return bcs.end();   }

  private:
    static BuoySupport *bs;

    bool fileRead;

    vec bcs;

    void * operator new(std::size_t n)
    {
      char * mem;
      PetscMalloc(n, &mem);
      return mem;
    }

    void operator delete(void * ptr)
    {
      PetscFree(ptr);
    }


    BuoySupport() : fileRead(false) { }
};

BuoySupport * BuoySupport::bs = 0;

void readBuoyBC(UserCtx * user) { BuoySupport::get()->readFile(user); }


// Sets the temperature buoyancy initial conditions
void buoyInit(UserCtx *user)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
            lxs, lxe, lys, lye, lzs, lze);

  // Setup access to temperature data.
  PetscReal ***ltmprt;
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  // Set temp in all cell to user->buoyTempRef
  PetscInt i, j, k;
  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  for (i=xs; i<xe; i++)
    ltmprt[k][j][i] = user->buoyTempRef;

  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);
  DMLocalToGlobalBegin(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  DMLocalToGlobalEnd(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  VecCopy(user->gTmprt, user->gTmprt_o);
}


#include "buoyancy.test.BC.h"

// Sets the temperature boundary conditions which happen
// to be inside the mesh.
void buoyTestBC(UserCtx *user)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  static bool once = true;

  // Setup access to temperature data.
  PetscReal ***ltmprt;
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

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
//       ltmprt[k][j][i] = user->buoyTemp;
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
        ltmprt[k][j][i] = user->buoyTemp;
      // printSequentialvaif(once && log, "k:  %4d  j:  %4d, i:  %4d\n", k, j, i);
    }
  }

  once = false;

  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);
  DMLocalToGlobalBegin(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  DMLocalToGlobalEnd(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  VecCopy(user->gTmprt, user->gTmprt_o);
}


#include "buoyancy.full.BC.h"


// Sets the temperature boundary conditions which happen
// to be inside the mesh.
void buoyMouthNoseBC(UserCtx *user)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  // printSequentialva("zs: %4d  ze: %4d  ys: %4d  ye: %4d  xs: %4d  xe: %4d\n",
  //                   zs, ze, ys, ye, xs, xe);

  // Setup access to temperature data.
  PetscReal ***ltmprt;
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  // Set temp in cells identified in buoyancy.full.BC.h.
  for (int imn=0; imn<mouthNoseBCArrayCnt; ++imn)
  {
    int k = mouthNoseBCArray[imn][0];
    int j = mouthNoseBCArray[imn][1];
    for (int i=mouthNoseBCArray[imn][2]; i<mouthNoseBCArray[imn][3]; ++i)
    {
      bool log = (k >= zs && k < ze && j >= ys && j < ye && i >= xs && i < xe);
      if (log)
        ltmprt[k][j][i] = user->buoyTemp;
      // printSequentialvaif(log, "k:  %4d  j:  %4d, i:  %4d\n", k, j, i);
    }
  }

  // TECOutVecRank("TmprtBC", "T", user, ltmprt);

  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);
  DMLocalToGlobalBegin(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  DMLocalToGlobalEnd(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  VecCopy(user->gTmprt, user->gTmprt_o);
}

void buoyFileBC(UserCtx *user)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  static bool once = true;

  // printSequentialva("zs: %4d  ze: %4d  ys: %4d  ye: %4d  xs: %4d  xe: %4d\n",
  //                   zs, ze, ys, ye, xs, xe);

  // Setup access to temperature data.
  PetscReal ***ltmprt;
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  // Set temp in cells identified in temp BC file.
  for (auto const & idxs : *BuoySupport::get())
  {
    int k = idxs[0];
    int j = idxs[1];
    for (int i=idxs[2]; i<idxs[3]; ++i)
    {
      bool log = (k >= zs && k < ze && j >= ys && j < ye && i >= xs && i < xe);
      // printSequentialvaif(once && log, "k:  %4d  j:  %4d, i:  %4d\n", k, j, i);
      if (log)
        ltmprt[k][j][i] = user->buoyTemp;
    }
  }
  once = false;


  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);
  DMLocalToGlobalBegin(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  DMLocalToGlobalEnd(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  VecCopy(user->gTmprt, user->gTmprt_o);
}


void RHS_Tmprt(UserCtx *user, Vec Tmprt_RHS)
{
  DM              da = user->da, fda = user->fda, fda2 = user->fda2;
  DMDALocalInfo   info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k;

  PetscReal       ***aj;

  PetscInt        lxs, lxe, lys, lye, lzs, lze;

  PetscReal       ***ltmprt, ***tmprt_o, ***tmprt_rhs;
  Cmpnts  ***ucont, ***ucat;
  Cmpnts  ***csi, ***eta, ***zet;
  PetscReal       ***nvert, ***distance, ***lf1, ***lnu_t, ***pr_t;

  Vec Fp1, Fp2, Fp3;
  PetscReal ***fp1, ***fp2, ***fp3;
  PetscReal ***visc1, ***visc2, ***visc3;

  Cmpnts  ***icsi, ***ieta, ***izet;
  Cmpnts  ***jcsi, ***jeta, ***jzet;
  Cmpnts  ***kcsi, ***keta, ***kzet;
  PetscReal       ***iaj, ***jaj, ***kaj, ***rho, ***mu;

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

  VecDuplicate(user->lTmprt, &Fp1);
  VecDuplicate(user->lTmprt, &Fp2);
  VecDuplicate(user->lTmprt, &Fp3);

  VecSet(Fp1,0);
  VecSet(Fp2,0);
  VecSet(Fp3,0);
  VecSet(user->lVisc1_tmprt,0);
  VecSet(user->lVisc2_tmprt,0);
  VecSet(user->lVisc3_tmprt,0);


  if (levelset)
  {
    DMDAVecGetArray(da, user->lDensity, &rho);
    DMDAVecGetArray(da, user->lMu, &mu);
  }

  //DMDAVecGetArray(da, user->lSrans, &lS);
  if (les_prt) DMDAVecGetArray(da, user->lNu_t, &lnu_t);

  if (les_prt) DMDAVecGetArray(da, user->lPr_t, &pr_t);


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

  DMDAVecGetArray(da, user->lTmprt, &ltmprt);
  DMDAVecGetArray(da, user->lTmprt_o, &tmprt_o);
  DMDAVecGetArray(da, Tmprt_RHS, &tmprt_rhs);

  DMDAVecGetArray(da, Fp1, &fp1);
  DMDAVecGetArray(da, Fp2, &fp2);
  DMDAVecGetArray(da, Fp3, &fp3);
  DMDAVecGetArray(da, user->lVisc1_tmprt, &visc1);
  DMDAVecGetArray(da, user->lVisc2_tmprt, &visc2);
  DMDAVecGetArray(da, user->lVisc3_tmprt, &visc3);


  double nu_t, D_m, D_t;

  for (k=lzs; k<lze; k++)
    for (j=lys; j<lye; j++)
      for (i=lxs-1; i<lxe; i++)
      {
        double csi0 = icsi[k][j][i].x,
               csi1 = icsi[k][j][i].y,
               csi2 = icsi[k][j][i].z;
        double eta0 = ieta[k][j][i].x,
               eta1 = ieta[k][j][i].y,
               eta2 = ieta[k][j][i].z;
        double zet0 = izet[k][j][i].x,
               zet1 = izet[k][j][i].y,
               zet2 = izet[k][j][i].z;
        double ajc = iaj[k][j][i];

        double g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
        double g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
        double g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

        double dtdc, dtde, dtdz;


        dtdc = ltmprt[k][j][i+1] - ltmprt[k][j][i];

        if ((nvert[k][j+1][i]) > 1.1 || (nvert[k][j+1][i+1]) > 1.1)
        {
          dtde = (ltmprt[k][j  ][i+1] + ltmprt[k][j  ][i]
                - ltmprt[k][j-1][i+1] - ltmprt[k][j-1][i]) * 0.5;
        }
        else if ((nvert[k][j-1][i]) > 1.1 || (nvert[k][j-1][i+1]) > 1.1)
        {
          dtde = (ltmprt[k][j+1][i+1] + ltmprt[k][j+1][i]
                - ltmprt[k][j  ][i+1] - ltmprt[k][j  ][i]) * 0.5;
        }
        else
        {
          dtde = (ltmprt[k][j+1][i+1] + ltmprt[k][j+1][i]
                - ltmprt[k][j-1][i+1] - ltmprt[k][j-1][i]) * 0.25;
        }

        if ((nvert[k+1][j][i]) > 1.1 || (nvert[k+1][j][i+1]) > 1.1)
        {
          dtdz = (ltmprt[k  ][j][i+1] + ltmprt[k  ][j][i]
                - ltmprt[k-1][j][i+1] - ltmprt[k-1][j][i]) * 0.5;

        }
        else if ((nvert[k-1][j][i]) > 1.1 || (nvert[k-1][j][i+1]) > 1.1)
        {
          dtdz = (ltmprt[k+1][j][i+1] + ltmprt[k+1][j][i]
                - ltmprt[k  ][j][i+1] - ltmprt[k  ][j][i]) * 0.5;
        }
        else {
          dtdz = (ltmprt[k+1][j][i+1] + ltmprt[k+1][j][i]
                - ltmprt[k-1][j][i+1] - ltmprt[k-1][j][i]) * 0.25;
        }


        D_t = 0.0;
        if (les_prt)
        {
          if ( i==0 || nvert[k][j][i] > 0.5 )
            D_t = pr_t[k][j][i+1];
          else if ( i==mx-2 || nvert[k][j][i+1] > 0.5 )
            D_t = pr_t[k][j][i];
          else
            D_t = 0.5 * (pr_t[k][j][i] + pr_t[k][j][i+1]);
        }

        fp1[k][j][i] = - ucont[k][j][i].x * 0.5
                     * ( ltmprt[k][j][i] + ltmprt[k][j][i+1]);

        D_m = 1./ (user->ren * user->Pr);


        visc1[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz)
                        * ajc * (D_m + D_t);
      }

  for (k=lzs; k<lze; k++)
    for (j=lys-1; j<lye; j++)
      for (i=lxs; i<lxe; i++)
      {
        double csi0 = jcsi[k][j][i].x,
               csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
        double eta0 = jeta[k][j][i].x,
               eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
        double zet0 = jzet[k][j][i].x,
               zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;
        double ajc = jaj[k][j][i];

        double g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
        double g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
        double g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

        double dtdc, dtde, dtdz;

        if ((nvert[k][j][i+1]) > 1.1 || (nvert[k][j+1][i+1]) > 1.1)
        {
          dtdc = (ltmprt[k][j+1][i  ] + ltmprt[k][j][i  ]
                - ltmprt[k][j+1][i-1] - ltmprt[k][j][i-1]) * 0.5;
        }
        else if ((nvert[k][j][i-1]) > 1.1 || (nvert[k][j+1][i-1]) > 1.1)
        {
          dtdc = (ltmprt[k][j+1][i+1] + ltmprt[k][j][i+1]
                - ltmprt[k][j+1][i  ] - ltmprt[k][j][i  ]) * 0.5;
        }
        else
        {
          dtdc = (ltmprt[k][j+1][i+1] + ltmprt[k][j][i+1]
                - ltmprt[k][j+1][i-1] - ltmprt[k][j][i-1]) * 0.25;
        }

        dtde = ltmprt[k][j+1][i] - ltmprt[k][j][i];

        if ((nvert[k+1][j][i]) > 1.1 || (nvert[k+1][j+1][i]) > 1.1)
        {
          dtdz = (ltmprt[k  ][j+1][i] + ltmprt[k  ][j][i]
                - ltmprt[k-1][j+1][i] - ltmprt[k-1][j][i]) * 0.5;
        }
        else if ((nvert[k-1][j][i]) > 1.1 || (nvert[k-1][j+1][i]) > 1.1)
        {
          dtdz = (ltmprt[k+1][j+1][i] + ltmprt[k+1][j][i]
                - ltmprt[k  ][j+1][i] - ltmprt[k  ][j][i]) * 0.5;
        }
        else
        {
          dtdz = (ltmprt[k+1][j+1][i] + ltmprt[k+1][j][i]
                - ltmprt[k-1][j+1][i] - ltmprt[k-1][j][i]) * 0.25;
        }


        D_t = 0.0;
        if (les_prt)
        {
          if ( j==0 || nvert[k][j][i] > 0.5)
            D_t = pr_t[k][j+1][i];
          else if ( j==my-2 || nvert[k][j+1][i] > 0.5)
            D_t = pr_t[k][j][i];
          else
            D_t = 0.5 * (pr_t[k][j][i] + pr_t[k][j+1][i]);
        }

        fp2[k][j][i] = -ucont[k][j][i].y * 0.5
                     * ( ltmprt[k][j][i] + ltmprt[k][j+1][i]);

        D_m = 1./(user->ren * user->Pr);

        visc2[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz) * ajc * (D_m + D_t);
      }

  for (k=lzs-1; k<lze; k++)
    for (j=lys; j<lye; j++)
      for (i=lxs; i<lxe; i++)
      {
        double csi0 = kcsi[k][j][i].x, csi1 = kcsi[k][j][i].y,
               csi2 = kcsi[k][j][i].z;
        double eta0 = keta[k][j][i].x, eta1 = keta[k][j][i].y,
               eta2 = keta[k][j][i].z;
        double zet0 = kzet[k][j][i].x, zet1 = kzet[k][j][i].y,
               zet2 = kzet[k][j][i].z;
        double ajc = kaj[k][j][i];

        double g11 = csi0 * zet0 + csi1 * zet1 + csi2 * zet2;
        double g21 = eta0 * zet0 + eta1 * zet1 + eta2 * zet2;
        double g31 = zet0 * zet0 + zet1 * zet1 + zet2 * zet2;

        double dtdc, dtde, dtdz;
        double nu_t, prt;

        if ((nvert[k][j][i+1])> 1.1 || (nvert[k+1][j][i+1])> 1.1)
        {
          dtdc = (ltmprt[k+1][j][i  ] + ltmprt[k][j][i  ]
                - ltmprt[k+1][j][i-1] - ltmprt[k][j][i-1]) * 0.5;
        }
        else if ((nvert[k][j][i-1])> 1.1 || (nvert[k+1][j][i-1])> 1.1)
        {
          dtdc = (ltmprt[k+1][j][i+1] + ltmprt[k][j][i+1]
                - ltmprt[k+1][j][i  ] - ltmprt[k][j][i  ]) * 0.5;
        }
        else
        {
          dtdc = (ltmprt[k+1][j][i+1] + ltmprt[k][j][i+1]
                - ltmprt[k+1][j][i-1] - ltmprt[k][j][i-1]) * 0.25;
        }

        if ((nvert[k][j+1][i])> 1.1 || (nvert[k+1][j+1][i])> 1.1)
        {
          dtde = (ltmprt[k+1][j  ][i] + ltmprt[k][j  ][i]
                - ltmprt[k+1][j-1][i] - ltmprt[k][j-1][i]) * 0.5;
        }
        else if ((nvert[k][j-1][i])> 1.1 || (nvert[k+1][j-1][i])> 1.1)
        {
          dtde = (ltmprt[k+1][j+1][i] + ltmprt[k][j+1][i]
                - ltmprt[k+1][j  ][i] - ltmprt[k][j  ][i]) * 0.5;
        }
        else
        {
          dtde = (ltmprt[k+1][j+1][i] + ltmprt[k][j+1][i]
                - ltmprt[k+1][j-1][i] - ltmprt[k][j-1][i]) * 0.25;
        }

        dtdz = ltmprt[k+1][j][i] - ltmprt[k][j][i];


        D_t = 0.0;
        if (les_prt)
        {
          if ( k==0 || nvert[k][j][i]>0.5)
            D_t = pr_t[k+1][j][i];
          else if ( k==mz-2 || nvert[k+1][j][i]>0.5)
            D_t = pr_t[k][j][i];
          else
            D_t = 0.5 * (pr_t[k][j][i] + pr_t[k+1][j][i]);
        }


        fp3[k][j][i] = -ucont[k][j][i].z * 0.5
                     * ( ltmprt[k][j][i] + ltmprt[k+1][j][i]);

        D_m = 1./ (user->ren * user->Pr);


        visc3[k][j][i] = (g11 * dtdc + g21 * dtde + g31 * dtdz)
                        * ajc * (D_m + D_t);
      }

  DMDAVecRestoreArray(da, Fp1, &fp1);
  DMDAVecRestoreArray(da, Fp2, &fp2);
  DMDAVecRestoreArray(da, Fp3, &fp3);
  DMDAVecRestoreArray(da, user->lVisc1_tmprt, &visc1);
  DMDAVecRestoreArray(da, user->lVisc2_tmprt, &visc2);
  DMDAVecRestoreArray(da, user->lVisc3_tmprt, &visc3);

  DMDALocalToLocalBegin(da, Fp1, INSERT_VALUES, Fp1);
  DMDALocalToLocalEnd(da, Fp1, INSERT_VALUES, Fp1);

  DMDALocalToLocalBegin(da, Fp2, INSERT_VALUES, Fp2);
  DMDALocalToLocalEnd(da, Fp2, INSERT_VALUES, Fp2);

  DMDALocalToLocalBegin(da, Fp3, INSERT_VALUES, Fp3);
  DMDALocalToLocalEnd(da, Fp3, INSERT_VALUES, Fp3);

  DMDALocalToLocalBegin(da, user->lVisc1_tmprt,
                                            INSERT_VALUES, user->lVisc1_tmprt);
  DMDALocalToLocalEnd(da, user->lVisc1_tmprt,
                                            INSERT_VALUES, user->lVisc1_tmprt);

  DMDALocalToLocalBegin(da, user->lVisc2_tmprt,
                                            INSERT_VALUES, user->lVisc2_tmprt);
  DMDALocalToLocalEnd(da, user->lVisc2_tmprt,
                                            INSERT_VALUES, user->lVisc2_tmprt);

  DMDALocalToLocalBegin(da, user->lVisc3_tmprt,
                                            INSERT_VALUES, user->lVisc3_tmprt);
  DMDALocalToLocalEnd(da, user->lVisc3_tmprt,
                                            INSERT_VALUES, user->lVisc3_tmprt);

  DMDAVecGetArray(da, Fp1, &fp1);
  DMDAVecGetArray(da, Fp2, &fp2);
  DMDAVecGetArray(da, Fp3, &fp3);
  DMDAVecGetArray(da, user->lVisc1_tmprt, &visc1);
  DMDAVecGetArray(da, user->lVisc2_tmprt, &visc2);
  DMDAVecGetArray(da, user->lVisc3_tmprt, &visc3);

  if (periodic)
    for (k=zs; k<ze; k++)
      for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++)
        {
          int a=i, b=j, c=k;

          int flag=0;

          if (i_periodic && i==0)          a=mx-2, flag=1;
          else if (i_periodic && i==mx-1)  a=1,    flag=1;

          if (j_periodic && j==0)          b=my-2, flag=1;
          else if (j_periodic && j==my-1)  b=1,    flag=1;

          if (k_periodic && k==0)          c=mz-2, flag=1;
          else if (k_periodic && k==mz-1)  c=1,    flag=1;

          if (ii_periodic && i==0)         a=-2,   flag=1;
          else if (ii_periodic && i==mx-1) a=mx+1, flag=1;

          if (jj_periodic && j==0)         b=-2,   flag=1;
          else if (jj_periodic && j==my-1) b=my+1, flag=1;

          if (kk_periodic && k==0)         c=-2,   flag=1;
          else if (kk_periodic && k==mz-1) c=mz+1, flag=1;

          if (flag)
          {
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
      for (i=lxs; i<lxe; i++)
      {
        if ( i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ||
             nvert[k][j][i]>0.1 )
        {
          tmprt_rhs[k][j][i] = 0.0 ;
          continue;
        }

        double ajc = aj[k][j][i];//, d = distance[k][j][i];

        if ( nvert[k][j][i] < 0.5 )
        {
          double r = 1.;

          if (levelset) r = rho[k][j][i];

          // advection
          double advection = (fp1[k][j][i] - fp1[k][j][i-1]
                            + fp2[k][j][i] - fp2[k][j-1][i]
                            + fp3[k][j][i] - fp3[k-1][j][i] ) * ajc;

          // diffusion
          double diffusion = (visc1[k][j][i] - visc1[k][j][i-1]
                            + visc2[k][j][i] - visc2[k][j-1][i]
                            + visc3[k][j][i] - visc3[k-1][j][i]) * ajc / r;
          tmprt_rhs[k][j][i] = advection + diffusion;
        }
      }


  // TECOutVecRank("TmpRHS", "rhs", user, tmprt_rhs);



  if (les_prt) DMDAVecRestoreArray(da, user->lNu_t, &lnu_t);
  if (les_prt) DMDAVecRestoreArray(da, user->lPr_t, &pr_t);

  DMDAVecRestoreArray(da, Fp1, &fp1);
  DMDAVecRestoreArray(da, Fp2, &fp2);
  DMDAVecRestoreArray(da, Fp3, &fp3);
  DMDAVecRestoreArray(da, user->lVisc1_tmprt, &visc1);
  DMDAVecRestoreArray(da, user->lVisc2_tmprt, &visc2);
  DMDAVecRestoreArray(da, user->lVisc3_tmprt, &visc3);
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

  DMDAVecRestoreArray(da, user->lTmprt, &ltmprt);
  DMDAVecRestoreArray(da, user->lTmprt_o, &tmprt_o);
  DMDAVecRestoreArray(da, Tmprt_RHS, &tmprt_rhs);

  if (levelset)
  {
    DMDAVecRestoreArray(da, user->lDensity, &rho);
    DMDAVecRestoreArray(da, user->lMu, &mu);
  }

  VecDestroy(&Fp1);
  VecDestroy(&Fp2);
  VecDestroy(&Fp3);

  // PetscPrintf(PETSC_COMM_WORLD, "Leaving RHS_Tmprt\n");
};


void Tmprt_IC(UserCtx *user)
{
  DMDALocalInfo   info;
  PetscInt        i, j, k;
  double nu = 1./(user->ren * user->Pr);

  PetscReal       ***ltmprt;
  PetscReal       ***nvert;

  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  DMDAVecGetArray(user->da, user->lNvert, &nvert);
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  Cmpnts          ***coor;
  Vec             Coor;

  DMDAGetCoordinates(user->da, &Coor);

  DMDAVecGetArray(user->fda, Coor, &coor);


  for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      { // pressure node; cell center

        //ltmprt[k][j][i] = coor[k][j][i].y*user->tmprts[3];
        ltmprt[k][j][i] = user->temp_IC;

        if (initial_perturbation)
        {
          double F = 1.0;                        // 3% noise
          int n = rand() % 20000;         // RAND_MAX = 65535
          n -= 10000;
//        ltmprt[k][j][i] *= 0.0;
//                          ( 1.0 + ((double)n)/10000. * F ); ??
//                          uin * (1+-0.xx) ?? which one??
        }



        if ( nvert[k][j][i]>0.5)
        {
//        ltmprt[k][j][i] = 0.0;
        }
        else if (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 )
        {
//        ltmprt[k][j][i] = 0.0;
        }


      }

  DMDAVecRestoreArray(user->fda, Coor, &coor);
  DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);

//  Tmprt_BC(user);

  DMLocalToGlobalBegin(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  DMLocalToGlobalEnd(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  VecCopy(user->gTmprt, user->gTmprt_o);


  if (user->buoyTest || user->buoyMouthNose || user->buoyFilenameSet)
  {
    buoyInit(user);
  }
};


void Force_Tmprt(UserCtx *user)
{
  DM              da = user->da;
  DMDALocalInfo   info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k;
  double nu = 1./(user->ren * user->Pr);
  double Ri_x = user->Ri_x;
  double Ri_y = user->Ri_y;
  double Ri_z = user->Ri_z;


  PetscInt        lxs, lxe, lys, lye, lzs, lze;


  PetscReal       ***ltmprt;
  PetscReal       ***nvert;

  Cmpnts          ***ftmprt, ***csi, ***eta, ***zet;

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
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);
  DMDAVecGetArray(user->fda, user->lFTmprt, &ftmprt);
  DMDAVecGetArray(user->fda, user->lCsi,  &csi);
  DMDAVecGetArray(user->fda, user->lEta,  &eta);
  DMDAVecGetArray(user->fda, user->lZet,  &zet);

//      PetscReal tmprt_xzAvg[my];

//      for (j=0; j<my; j++)
//      {
//              tmprt_xzAvg[j] = 0.0;
//      }


//        for (k=lzs; k<lze; k++)
//        for (j=lys; j<lye; j++)
//        for (i=lxs; i<lxe; i++)
//          pressure node; cell center
//        {

//              tmprt_xzAvg[j]+=ltmprt[k][j][i];

//      }


//      PetscBarrier(PETSC_NULL);

//      PetscReal     u_sum;
//      for (j=0; j<my; j++)
//      {

//              PetscGlobalSum(&(tmprt_xzAvg[j]), &u_sum, PETSC_COMM_WORLD);
//                      tmprt_xzAvg[j] = u_sum/((double)(mx*mz));

//              printf("the avg temperature %d %le \n", j, tmprt_xzAvg[j]);
//      }



  double force_x, force_y, force_z;

  for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      {

        // Used for dimensional runs
        if (user->buoyMouthNose || user->buoyTest)
        {
          force_x = 0.0;
          force_y = 0.0;
          force_z = (1.0 - user->buoyBeta * (ltmprt[k][j][i] - user->buoyTempRef))
                  * -gravity_z;
        }

        // Used for non-dimensional runs with the Richardson number
        else
        {
          // force_x = Ri_x * ltmprt[k][j][i];
          // force_y = Ri_y * ltmprt[k][j][i];
          // force_z = Ri_z * ltmprt[k][j][i];

          double T = 1 - user->buoyTempRef / ltmprt[k][j][i];
          force_x = Ri_x * T;
          force_y = Ri_y * T;
          force_z = Ri_z * T;
        }

        ftmprt[k][j][i].x = force_x * csi[k][j][i].x
                          + force_y * csi[k][j][i].y
                          + force_z * csi[k][j][i].z;
        ftmprt[k][j][i].y = force_x * eta[k][j][i].x
                          + force_y * eta[k][j][i].y
                          + force_z * eta[k][j][i].z;
        ftmprt[k][j][i].z = force_x * zet[k][j][i].x
                          + force_y * zet[k][j][i].y
                          + force_z * zet[k][j][i].z;


        if (nvert[k][j][i]>0.5)
        {
          ftmprt[k][j][i].x = 0.0;
          ftmprt[k][j][i].y = 0.0;
          ftmprt[k][j][i].z = 0.0;
        }
//      else if (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1||j==my-2)
//      {
//        ftmprt[k][j][i].x = 0.0;
//        ftmprt[k][j][i].y = 0.0;
//        ftmprt[k][j][i].z = 0.0;
//      }

//      if (user->bctype_ltmprt[1] && i==mx-2) ftmprt[k][j][i].x = 0.0;
//      if (user->bctype_ltmprt[3] && j==my-2) ftmprt[k][j][i].y = 0.0;
//      if (user->bctype_ltmprt[5] && k==mz-2) ftmprt[k][j][i].z = 0.0;


      }


  if (xs == 0)
  {
    i = 0;
    for (k=zs; k<ze; k++)
      for (j=ys; j<ye; j++)
      {
        ftmprt[k][j][i].x = 0;
        ftmprt[k][j][i].y = 0;
        ftmprt[k][j][i].z = 0;
      }
  }

  if (xe == mx)
  {
    for (k=zs; k<ze; k++)
      for (j=ys; j<ye; j++)
      {
        if (!i_periodic && !ii_periodic)
        {
          i = mx-2;
          ftmprt[k][j][i].x = 0;
        }
        i = mx-1;
        ftmprt[k][j][i].x = 0;
        ftmprt[k][j][i].y = 0;
        ftmprt[k][j][i].z = 0;
      }
  }


  if (ys == 0)
  {
    for (k=zs; k<ze; k++)
      for (i=xs; i<xe; i++)
      {
        j=0;
        ftmprt[k][j][i].x = 0;
        ftmprt[k][j][i].y = 0;
        ftmprt[k][j][i].z = 0;
      }
  }

  if (ye == my)
  {
    for (k=zs; k<ze; k++)
      for (i=xs; i<xe; i++)
      {
        if (!j_periodic && !jj_periodic)
        {
          j=my-2;
          ftmprt[k][j][i].y = 0;
        }
        j=my-1;
        ftmprt[k][j][i].x = 0;
        ftmprt[k][j][i].y = 0;
        ftmprt[k][j][i].z = 0;
      }
  }


  if (zs == 0)
  {
    k=0;
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      {
        ftmprt[k][j][i].x = 0;
        ftmprt[k][j][i].y = 0;
        ftmprt[k][j][i].z = 0;
      }
  }

  if (ze == mz)
  {
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      {
        if (!k_periodic && !kk_periodic)
        {
          k=mz-2;
          ftmprt[k][j][i].z = 0;
        }
        k=mz-1;
        ftmprt[k][j][i].x = 0;
        ftmprt[k][j][i].y = 0;
        ftmprt[k][j][i].z = 0;
      }
  }


  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);
  DMDAVecRestoreArray(user->fda, user->lFTmprt, &ftmprt);

  DMLocalToGlobalBegin(user->fda, user->lFTmprt, INSERT_VALUES, user->gFTmprt);
  DMLocalToGlobalEnd(user->fda, user->lFTmprt, INSERT_VALUES, user->gFTmprt);

//  TECIOOut_rhs(user,  user->FTmprt);

  DMDAVecRestoreArray(user->fda, user->lCsi,  &csi);
  DMDAVecRestoreArray(user->fda, user->lEta,  &eta);
  DMDAVecRestoreArray(user->fda, user->lZet,  &zet);

};


void Tmprt_BC(UserCtx *user)
{
  DM              da = user->da, fda = user->fda;
  DMDALocalInfo   info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k, ibi;
  //double nu = 1./user->ren;
  PetscReal       ***aj, ***distance, ***rho, ***mu;

  PetscInt        lxs, lxe, lys, lye, lzs, lze;


  PetscReal ***ltmprt;
  Cmpnts  ***csi, ***eta, ***zet, ***ucat;
  PetscReal       ***nvert, ***ustar;


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

  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(da, user->lAj, &aj);
  DMDAVecGetArray(da, user->lUstar, &ustar);

  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  if (levelset)
  {
    DMDAVecGetArray(da, user->lDensity, &rho);
    DMDAVecGetArray(da, user->lMu, &mu);
  }


  // BC for temperature
  for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      {
        // pressure node; cell center
        double ren = user->ren;
        //if (levelset) ren = rho[k][j][i]/ mu[k][j][i];

        // from saved inflow file
        if (inletprofile==100 && k==1)
        {
          ltmprt[k-1][j][i] = user->tmprt_plane[j][i];
        }
        // inflow
        else if ( user->bctype_tmprt[4]==5 && k==1 && nvert[k][j][i]<0.5)
        {
          ltmprt[k-1][j][i] =2.0*user->tmprts[4]-ltmprt[k][j][i];
        }

        // outflow
        if ( user->bctype_tmprt[5]==4 && k==mz-2)
        {
          ltmprt[k+1][j][i] =ltmprt[k][j][i];
        }
        // NO energy flux
        if ( user->bctype_tmprt[0] == 0 && i==1 )
          ltmprt[k][j][i-1] = ltmprt[k][j][i];
        if ( user->bctype_tmprt[1] == 0 && i==mx-2 )
          ltmprt[k][j][i+1] = ltmprt[k][j][i];
        if ( user->bctype_tmprt[2] == 0 && j==1 )
          ltmprt[k][j-1][i] = ltmprt[k][j][i];
        if ( user->bctype_tmprt[3] == 0 && j==my-2 )
          ltmprt[k][j+1][i] = ltmprt[k][j][i];
        if ( user->bctype_tmprt[4] == 0 && k==1 )
          ltmprt[k-1][j][i] = ltmprt[k][j][i];
        if ( user->bctype_tmprt[5] == 0 && k==mz-2 )
          ltmprt[k+1][j][i] = ltmprt[k][j][i];

        // fixed temperature

        if (user->bctype_tmprt[0] == 1 && i==1)
        {
          ltmprt[k][j][i-1] = ltmprt[k][j][i] = user->tmprts[0];
        }

        if (user->bctype_tmprt[1] == 1 && i==mx-2 )
        {
          ltmprt[k][j][i+1] = ltmprt[k][j][i] = user->tmprts[1];
        }

        if (user->bctype_tmprt[2] == 1 && j==1)
        {
          ltmprt[k][j-1][i] = ltmprt[k][j][i] = user->tmprts[2];
        }

        if (user->bctype_tmprt[3] == 1 && j==my-2 )
        {
          ltmprt[k][j+1][i] = ltmprt[k][j][i] = user->tmprts[3];
        }

        if (user->bctype_tmprt[4] == 1 && k==1    )
        {
          ltmprt[k-1][j][i] = ltmprt[k][j][i] = user->tmprts[4];
        }

        if (user->bctype_tmprt[5] == 1 && k==mz-2 )
        {
          ltmprt[k+1][j][i] = ltmprt[k][j][i] = user->tmprts[5];
        }


        // Uniform heat flux

        double D_m = 1./ (user->ren * user->Pr);
        if ( user->bctype_tmprt[0] == 2 && i==1 )
        {
          double area = sqrt( csi[k][j][i].x*csi[k][j][i].x
                            + csi[k][j][i].y*csi[k][j][i].y
                            + csi[k][j][i].z*csi[k][j][i].z );
          double hx = 1.0/aj[k][j][i]/area;
          ltmprt[k][j][i-1] = -user->tmprts[0]/D_m + ltmprt[k][j][i];
        }

        if ( user->bctype_tmprt[1] == 2 && i==mx-2 )
        {
          double area = sqrt( csi[k][j][i].x*csi[k][j][i].x
                            + csi[k][j][i].y*csi[k][j][i].y
                            + csi[k][j][i].z*csi[k][j][i].z );
          double hx = 1.0/aj[k][j][i]/area;
          ltmprt[k][j][i+1] =  user->tmprts[1]/D_m + ltmprt[k][j][i];
        }


        if ( user->bctype_tmprt[2] == 2 && j==1 )
        {
          double area = sqrt( eta[k][j][i].x*eta[k][j][i].x
                            + eta[k][j][i].y*eta[k][j][i].y
                            + eta[k][j][i].z*eta[k][j][i].z );
          double hx = 1.0/aj[k][j][i]/area;
          ltmprt[k][j-1][i] = -user->tmprts[2]/D_m + ltmprt[k][j][i];
        }

        if ( abs(user->bctype_tmprt[3]) == 2 && j==my-2 )
        {
          double area = sqrt( eta[k][j][i].x*eta[k][j][i].x
                            + eta[k][j][i].y*eta[k][j][i].y
                            + eta[k][j][i].z*eta[k][j][i].z );
          double hx = 1.0/aj[k][j][i]/area;
          ltmprt[k][j+1][i] = user->tmprts[3]/D_m + ltmprt[k][j][i];
        }


        if ( abs(user->bctype_tmprt[4]) == 2 && k==1 )
        {
          double area = sqrt( zet[k][j][i].x*zet[k][j][i].x
                            + zet[k][j][i].y*zet[k][j][i].y
                            + zet[k][j][i].z*zet[k][j][i].z );
          double hx = 1.0/aj[k][j][i]/area;
          ltmprt[k-1][j][i] = -user->tmprts[4]/D_m + ltmprt[k][j][i];
        }

        if ( abs(user->bctype_tmprt[5]) == 2 && k==mz-2 )
        {
          double area = sqrt( zet[k][j][i].x*zet[k][j][i].x
                            + zet[k][j][i].y*zet[k][j][i].y
                            + zet[k][j][i].z*zet[k][j][i].z );
          double hx = 1.0/aj[k][j][i]/area;
          ltmprt[k+1][j][i] = user->tmprts[5]/D_m + ltmprt[k][j][i];
        }
      }

  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);

  DMDALocalToLocalBegin(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);
  DMDALocalToLocalEnd(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);

  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      {
        int flag=0, a=i, b=j, c=k;

        if (i_periodic && i==0)          a=mx-2, flag=1;
        else if (i_periodic && i==mx-1)  a=1, flag=1;

        if (j_periodic && j==0)          b=my-2, flag=1;
        else if (j_periodic && j==my-1)  b=1, flag=1;

        if (k_periodic && k==0)          c=mz-2, flag=1;
        else if (k_periodic && k==mz-1)  c=1, flag=1;


        if (ii_periodic && i==0)         a=-2, flag=1;
        else if (ii_periodic && i==mx-1) a=mx+1, flag=1;

        if (jj_periodic && j==0)         b=-2, flag=1;
        else if (jj_periodic && j==my-1) b=my+1, flag=1;

        if (kk_periodic && k==0)         c=-2, flag=1;
        else if (kk_periodic && k==mz-1) c=mz+1, flag=1;

        if (flag) ltmprt[k][j][i] = ltmprt[c][b][a];
      }


  DMDAVecRestoreArray(fda, user->lCsi, &csi);
  DMDAVecRestoreArray(fda, user->lEta, &eta);
  DMDAVecRestoreArray(fda, user->lZet, &zet);
  DMDAVecRestoreArray(fda, user->lUcat, &ucat);

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(da, user->lAj, &aj);
  DMDAVecRestoreArray(da, user->lUstar, &ustar);

  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);

  if (levelset)
  {
    DMDAVecRestoreArray(da, user->lDensity, &rho);
    DMDAVecRestoreArray(da, user->lMu, &mu);
  }

  DMDALocalToLocalBegin(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);
  DMDALocalToLocalEnd(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);


  if (user->buoyTest)
    buoyTestBC(user);

  if (user->buoyMouthNose)
    buoyMouthNoseBC(user);

  if (user->buoyFilenameSet)
    buoyFileBC(user);
};

PetscErrorCode FormFunction_Tmprt(SNES snes, Vec Tmprt, Vec Rhs, void *ptr)
{
  // PetscPrintf(PETSC_COMM_WORLD, "Entering FormFunction_Tmprt\n");

  UserCtx *user = (UserCtx*)ptr;

  DMDALocalInfo   info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k;
  PetscInt        lxs, lxe, lys, lye, lzs, lze;

  PetscReal       ***nvert;
  PetscReal ***rhs;

  Vec Rhs_wm;

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


  DMGlobalToLocalBegin(user->da, Tmprt, INSERT_VALUES, user->lTmprt);
  DMGlobalToLocalEnd(user->da, Tmprt, INSERT_VALUES, user->lTmprt);


  Tmprt_BC(user);
  RHS_Tmprt(user, Rhs);

  if (user->temp && Shear_wm &&
        (imin_wmtmprt != 0 || imax_wmtmprt != 0 || jmin_wmtmprt != 0 ||
         jmax_wmtmprt != 0 || IB_wmtmprt != 0))
  {
    VecDuplicate(Rhs, &Rhs_wm);
    VecSet(Rhs_wm,0.0);
    Formfunction_wm_tmprt(user, Rhs_wm, 1.0);
    VecAXPY(Rhs, 1, Rhs_wm);
  }


  double coeff = time_coeff();


  if ( coeff>0.9 && coeff<1.1 )
  {
    VecAXPY(Rhs, -1/user->dt, Tmprt);
    VecAXPY(Rhs, 1/user->dt, user->gTmprt_o);

  }
  else/* if ( coeff > 1.4 && coeff < 1.6 )*/ {
    VecAXPY(Rhs, -1.5/user->dt, Tmprt);
    VecAXPY(Rhs, 2./user->dt, user->gTmprt_o);
    VecAXPY(Rhs, -0.5/user->dt, user->gTmprt_rm1);

  }


  DMDAVecGetArray(user->da, user->lNvert, &nvert);
  DMDAVecGetArray(user->da, Rhs, &rhs);

  for (k=zs; k<ze; k++)
    for (j=ys; j<ye; j++)
      for (i=xs; i<xe; i++)
      {


        if (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ||
            nvert[k][j][i]>0.5)
        {
          rhs[k][j][i] = 0;
        }


      }
  DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, Rhs, &rhs);

//  TECIOOut_rhs1(user, Rhs);
//  cout << "here 2\n";
//  cin >> aaa;


  if (user->temp && Shear_wm && (imin_wmtmprt != 0 || imax_wmtmprt != 0 ||
                                 jmin_wmtmprt != 0 || jmax_wmtmprt != 0 ||
                                 IB_wmtmprt != 0))
    VecDestroy(&Rhs_wm);

  // PetscPrintf(PETSC_COMM_WORLD, "Leaving FormFunction_Tmprt\n");

  return(0);
}

int snes_Tmprt_created=0;
Vec r_Tmprt;
Mat J_Tmprt;
SNES snes_Tmprt_eq;

void Solve_Tmprt(UserCtx *user, double & norm)
{

  KSP ksp;
  PC pc;

  int bi=0;
  double tol=1.e-6;//1.e-6

  if (!snes_Tmprt_created)
  {
    snes_Tmprt_created=1;

    VecDuplicate(user[bi].gTmprt, &r_Tmprt);
    SNESCreate(PETSC_COMM_WORLD,&snes_Tmprt_eq);

    PetscPrintf(PETSC_COMM_WORLD, "In Solve_Tmprt, calling SNESSetFunction\n");
    SNESSetFunction(snes_Tmprt_eq,r_Tmprt,FormFunction_Tmprt,(void *)&user[bi]);
    PetscPrintf(PETSC_COMM_WORLD, "In Solve_Tmprt, return from SNESSetFunction\n");

    MatCreateSNESMF(snes_Tmprt_eq, &J_Tmprt);
    SNESSetJacobian(snes_Tmprt_eq,J_Tmprt,J_Tmprt,MatMFFDComputeJacobian,
                    (void *)&user[bi]);
    SNESSetType(snes_Tmprt_eq, SNESTR);                     //SNESTR,SNESLS
    SNESSetMaxLinearSolveFailures(snes_Tmprt_eq,10000);
    SNESSetMaxNonlinearStepFailures(snes_Tmprt_eq,10000);
    SNESKSPSetUseEW(snes_Tmprt_eq, PETSC_TRUE);
    SNESKSPSetParametersEW(snes_Tmprt_eq,3,PETSC_DEFAULT,PETSC_DEFAULT,
                           PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                           PETSC_DEFAULT);
    SNESSetTolerances(snes_Tmprt_eq,PETSC_DEFAULT,tol,PETSC_DEFAULT,50,50000);

    SNESGetKSP(snes_Tmprt_eq, &ksp);
    KSPSetType(ksp, KSPGMRES);
    //KSPGMRESSetPreAllocateVectors(ksp);

    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCNONE);

    int maxits=50 /*20<- was the default on 2002-11-09 WRO  10000*/;
    double rtol=tol, atol=PETSC_DEFAULT, dtol=PETSC_DEFAULT;
    KSPSetTolerances(ksp,rtol,atol,dtol,maxits);
  }

  extern PetscErrorCode MySNESMonitor(SNES snes,PetscInt n,
                                      PetscReal rnorm,void *dummy);
  SNESMonitorSet(snes_Tmprt_eq,MySNESMonitor,PETSC_NULL,PETSC_NULL);

#if 0
  // temporary
  if (ti % 10 == 0)
  {
    PetscReal ***gtmprt;
    DMDAVecGetArray(user->da, user->gTmprt, &gtmprt);
    TECOutVec("gtmprt", user, gtmprt);
    DMDAVecRestoreArray(user->da, user->gTmprt, &gtmprt);
  }
#endif

  PetscPrintf(PETSC_COMM_WORLD, "\nSolving Temperature...\n");

  SNESSolve(snes_Tmprt_eq, PETSC_NULL, user[bi].gTmprt);

  SNESGetFunctionNorm(snes_Tmprt_eq, &norm);
  PetscPrintf(PETSC_COMM_WORLD, "\nTemperature SNES residual norm=%.5e\n\n",
              norm);

  DMDALocalInfo   info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k;
  PetscInt        lxs, lxe, lys, lye, lzs, lze;

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

  DMGlobalToLocalBegin(user->da, user->gTmprt, INSERT_VALUES, user->lTmprt);
  DMGlobalToLocalEnd(user->da, user->gTmprt, INSERT_VALUES, user->lTmprt);

  PetscReal ***gtmprt, ***ltmprt;

  DMDAVecGetArray(user->da, user->gTmprt, &gtmprt);
  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);

  if (periodic)
    for (k=zs; k<ze; k++)
      for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++)
        {
          int flag=0, a=i, b=j, c=k;

          if (i_periodic && i==0)          a=mx-2, flag=1;
          else if (i_periodic && i==mx-1)  a=1, flag=1;

          if (j_periodic && j==0)          b=my-2, flag=1;
          else if (j_periodic && j==my-1)  b=1, flag=1;

          if (k_periodic && k==0)          c=mz-2, flag=1;
          else if (k_periodic && k==mz-1)  c=1, flag=1;


          if (ii_periodic && i==0)         a=-2, flag=1;
          else if (ii_periodic && i==mx-1) a=mx+1, flag=1;

          if (jj_periodic && j==0)         b=-2, flag=1;
          else if (jj_periodic && j==my-1) b=my+1, flag=1;

          if (kk_periodic && k==0)         c=-2, flag=1;
          else if (kk_periodic && k==mz-1) c=mz+1, flag=1;

          if (flag) gtmprt[k][j][i] = ltmprt[c][b][a];
        }

  DMDAVecRestoreArray(user->da, user->gTmprt, &gtmprt);
  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);

  DMGlobalToLocalBegin(user->da, user->gTmprt, INSERT_VALUES, user->lTmprt);
  DMGlobalToLocalEnd(user->da, user->gTmprt, INSERT_VALUES, user->lTmprt);
  Tmprt_BC(user);
  DMLocalToGlobalBegin(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);
  DMLocalToGlobalEnd(user->da, user->lTmprt, INSERT_VALUES, user->gTmprt);


/*
  if (ti % 10 == 0)
  {
    DMDAVecGetArray(user->da, user->gTmprt, &gtmprt);
    TECOutVec("gtmprt", user, gtmprt);
    DMDAVecRestoreArray(user->da, user->gTmprt, &gtmprt);
  }
*/

  /*
    char fname[80];
    sprintf(fname,"Tmprt");
    TECIOOut_rhs_da(user, user->Tmprt, fname);
  */


//  TECIOOut_rhs1(user,  user->Tmprt_o);
//  TECIOOut_rhs1(user,  user->Tmprt);

};


PetscErrorCode TE_Output(UserCtx *user)
{
  DMDALocalInfo info = user->info;
  PetscInt        xs = info.xs, xe = info.xs + info.xm;
  PetscInt        ys = info.ys, ye = info.ys + info.ym;
  PetscInt        zs = info.zs, ze = info.zs + info.zm;
  PetscInt        mx = info.mx, my = info.my, mz = info.mz;

  PetscInt i, j, k;
  PetscInt lxs, lys, lzs, lxe, lye, lze;

  PetscReal ***ltmprt;
  PetscReal ***aj;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  DMDAVecGetArray(user->da, user->lTmprt, &ltmprt);
  DMDAVecGetArray(user->da, user->lAj, &aj);

  double local_sum=0, sum=0;
  for (k=lzs; k<lze; k++)
    for (j=lys; j<lye; j++)
      for (i=lxs; i<lxe; i++)
      {
        local_sum += 0.5 * ltmprt[k][j][i] * ltmprt[k][j][i] / aj[k][j][i];
      }
  MPI_Allreduce (&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  DMDAVecRestoreArray(user->da, user->lTmprt, &ltmprt);
  DMDAVecRestoreArray(user->da, user->lAj, &aj);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (!rank)
  {
    char filen[80];
    sprintf(filen, "%s/Tmperature_Square.dat", path);
    FILE *f = fopen(filen, "a");
    PetscFPrintf(PETSC_COMM_WORLD, f, "%d\t%.7e\n", ti, sum);
    fclose(f);
  }

  return 0;
}

// xyang add fluctuations
PetscErrorCode Add_fluc_tmprt(UserCtx *user)
{
  DM da = user->da, fda = user->fda;
  DMDALocalInfo   info = user->info;
  PetscInt        xs = info.xs, xe = info.xs + info.xm;
  PetscInt        ys = info.ys, ye = info.ys + info.ym;
  PetscInt        zs = info.zs, ze = info.zs + info.zm;
  PetscInt        mx = info.mx, my = info.my, mz = info.mz;
  PetscInt        lxs, lxe, lys, lye, lzs, lze;

  Cmpnts ***ucont, ***cent;
  Cmpnts ***icsi, ***jeta, ***kzet, ***zet;

  PetscInt i, j, k;

  PetscReal       ***nvert, ***p, ***level;       //seokkoo


  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  Vec Coor;
  Cmpnts  ***coor;
  DMDAGetGhostedCoordinates(da, &Coor);

  if (levelset) DMDAVecGetArray(da, user->lLevelset, &level);
  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(da, user->gTmprt, &p);


  srand( time(NULL)) ;    // seokkoo
  for (i = 0; i < (rand() % 3000); i++) (rand() % 3000);  //seokkoo

  if ( initial_perturbation)
  {
    PetscPrintf(PETSC_COMM_WORLD, "\nGenerating initial perturbation\n");
    for(k=lzs; k<lze; k++)
    {        // for all CPU
      for (j=lys; j<lye; j++)
        for (i=lxs; i<lxe; i++)
        {
          if (nvert[k][j][i]+nvert[k+1][j][i] < 0.5)
          {
            int n1, n2, n3;
            double F;

            F  = 0.1; // 100%
            n1 = rand() % 20000 - 10000;
            n2 = rand() % 20000 - 10000;
            n3 = rand() % 20000 - 10000;
            p[k][j][i] *= ( 1.0 + ((double)n3)/10000. * F );
          }
        }
    }
  }

  if (levelset) DMDAVecRestoreArray(da, user->lLevelset, &level);
  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(da, user->gTmprt, &p);

  DMGlobalToLocalBegin(da, user->gTmprt, INSERT_VALUES, user->lTmprt);
  DMGlobalToLocalEnd(da, user->gTmprt, INSERT_VALUES, user->lTmprt);


  return 0;
}


// Support routines for buoyancy calculation in its own namespace.
//namespace BouySupport
//{
//  bool readBuoyBCFile = false;
//}
