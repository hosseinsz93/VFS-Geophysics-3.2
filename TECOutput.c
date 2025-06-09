#include "variables.h"

#include "MeshParms.h"
#include "TECOutput.h"


void TECOutx(FILE * fo, int buffSize, double * x)
{
  int cnt = 0;
  for (int i=0; i<buffSize; ++i)
  {
    ++cnt;
    char const * format = cnt == 0 ? "%e" : " %e";
    fprintf(fo, format, x[i]);
    if (cnt == 10)
    {
      fprintf(fo, "\n");
      cnt = 0;
    }
  }
  if (cnt != 0)
    fprintf(fo, "\n");
}


void TECOutCoor(FILE * fo, UserCtx * user)
{
  // Set the mesh variables used in most routines.
  // to call:
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  Vec Coor;
  Cmpnts ***coor;
  DMDAGetCoordinates(user->da, &Coor);
  DMDAVecGetArray(user->fda, Coor, &coor);

  PetscInt IMax, JMax, KMax;
  IMax = mx - 1;
  JMax = my - 1;
  KMax = mz - 1;
  int buffSize = IMax * JMax * KMax;
  double * x = new double[buffSize];

  int i, j, k;

  for (k=zs; k<ze-1; k++)
  for (j=ys; j<ye-1; j++)
  for (i=xs; i<xe-1; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].x;
  TECOutx(fo, buffSize, x);

  for (k=zs; k<ze-1; k++)
  for (j=ys; j<ye-1; j++)
  for (i=xs; i<xe-1; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].y;
  TECOutx(fo, buffSize, x);

  for (k=zs; k<ze-1; k++)
  for (j=ys; j<ye-1; j++)
  for (i=xs; i<xe-1; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].z;
  TECOutx(fo, buffSize, x);

  delete [] x;

  DMDAVecRestoreArray(user->fda, Coor, &coor);
}

void TECOutCmpntsCell(char const * fname, FILE * fo, UserCtx * user, Cmpnts ***cellData)
{
  // Set the mesh variables used in most routines.
  // to call:
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
            lxs, lxe, lys, lye, lzs, lze);

  int IMax, JMax, KMax;
  IMax = mx - 2;
  JMax = my - 2;
  KMax = mz - 2;
  int buffSize = IMax * JMax * KMax;

  int i, j, k, cnt;

  double * x = new double[buffSize];
  memset(x, 0, buffSize&sizeof(double));

  for (k=zs; k<ze-2; k++)
  for (j=ys; j<ye-2; j++)
  for (i=xs; i<xe-2; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = cellData[k+1][j+1][i+1].x;
  TECOutx(fo, buffSize, x);

  for (k=zs; k<ze-2; k++)
  for (j=ys; j<ye-2; j++)
  for (i=xs; i<xe-2; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = cellData[k+1][j+1][i+1].y;
  TECOutx(fo, buffSize, x);

  for (k=zs; k<ze-2; k++)
  for (j=ys; j<ye-2; j++)
  for (i=xs; i<xe-2; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = cellData[k+1][j+1][i+1].z;
  TECOutx(fo, buffSize, x);

  delete [] x;
}

void TECOutVecCell(char const * fname, FILE * fo, UserCtx * user, double ***cellData)
{
  // Set the mesh variables used in most routines.
  // to call:
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
            lxs, lxe, lys, lye, lzs, lze);

  int IMax, JMax, KMax;
  IMax = mx - 2;
  JMax = my - 2;
  KMax = mz - 2;
  int buffSize = IMax * JMax * KMax;

  int i, j, k, cnt;

  double * x = new double[buffSize];
  memset(x, 0, buffSize&sizeof(double));

  for (k=zs; k<ze-2; k++)
  for (j=ys; j<ye-2; j++)
  for (i=xs; i<xe-2; i++)
    x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = cellData[k+1][j+1][i+1];
  TECOutx(fo, buffSize, x);

  delete [] x;
}


void TECOutCmpnts(char const * fname, UserCtx * user, Cmpnts ***ftmprt)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  char filen[80];
  sprintf(filen, "%s_%06d_%03d.plt", fname, ti, rank);

  FILE * fo;
  fo = fopen(filen, "w");

  fprintf(fo, "TITLE = \"%s\"\n", fname);
  // fprintf(fo, "FILETYPE = GRID\n");
  fprintf(fo, "VARIABLES = \"X\", \"Y\", \"Z\", \"Fx\", \"Fy\", \"Fz\"\n");
  // fprintf(fo, "VARIABLES = \"X\", \"Y\", \"Z\"\n");
  fprintf(fo, "ZONE, I=%d, J=%d, K=%d, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", mx-1, my-1, mz-1);
  // fprintf(fo, "ZONE, I=%d, J=%d, K=%d, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL)\n", mx-1, my-1, mz-1);

  TECOutCoor(fo, user);
  TECOutCmpntsCell(fname, fo, user, ftmprt);

  fclose(fo);
}


void TECOutVec(char const * fname, UserCtx * user, double ***vec)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  char filen[80];
  sprintf(filen, "%s_%06d_%03d.plt", fname, ti, rank);

  FILE * fo;
  fo = fopen(filen, "w");

  fprintf(fo, "TITLE = \"%s\"\n", fname);
  // fprintf(fo, "FILETYPE = GRID\n");
  fprintf(fo, "VARIABLES = \"X\", \"Y\", \"Z\", \"T\"\n");
  // fprintf(fo, "VARIABLES = \"X\", \"Y\", \"Z\"\n");
  fprintf(fo, "ZONE, I=%d, J=%d, K=%d, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED)\n", mx-1, my-1, mz-1);
  // fprintf(fo, "ZONE, I=%d, J=%d, K=%d, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL)\n", mx-1, my-1, mz-1);

  TECOutCoor(fo, user);
  TECOutVecCell(fname, fo, user, vec);

  fclose(fo);
}

//-----------------------------------------------------------------------------


void TECOutVecCellRank(char const * fname, FILE * fo, UserCtx * user, double ***cellData)
{
  // Set the mesh variables used in most routines.
  // to call:
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
            lxs, lxe, lys, lye, lzs, lze);

  int i, j, k, cnt;

  for (k=zs; k<ze; k++)
  for (j=ys; j<ye; j++)
  {
    fprintf(fo, "k,j,is-ie: %d %d %d-%d\n", k, j, xs, xe);
    for (i=xs; i<xe; i++)
      fprintf(fo, " %e", cellData[k][j][i]);
    fprintf(fo, "\n");
  }
}


void TECOutVecRank(char const * fname, char const * vname, UserCtx * user, double ***vec)
{
  PetscInt mx, my, mz; // Dimensions in three directions
  PetscInt xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt lxs, lxe, lys, lye, lzs, lze;
  meshParms(user, mx, my, mz, xs, xe, ys, ye, zs, ze,
                  lxs, lxe, lys, lye, lzs, lze);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  char filen[80];
  sprintf(filen, "%s_%06d_%03d.plt", fname, ti, rank);

  FILE * fo;
  fo = fopen(filen, "w");

  fprintf(fo, "TITLE = \"%s\"\n", fname);
  // fprintf(fo, "FILETYPE = GRID\n");
  fprintf(fo, "VARIABLES = \"X\", \"Y\", \"Z\", \"%s\"\n", vname);
  // fprintf(fo, "VARIABLES = \"X\", \"Y\", \"Z\"\n");
  fprintf(fo, "ZONE, I=%d, J=%d, K=%d, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED)\n", xe-xs, ye-ys, ze-zs);
  // fprintf(fo, "ZONE, I=%d, J=%d, K=%d, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL)\n", mx-1, my-1, mz-1);

  // TECOutCoorRank(fo, user);
  TECOutVecCellRank(fname, fo, user, vec);

  fclose(fo);
}
