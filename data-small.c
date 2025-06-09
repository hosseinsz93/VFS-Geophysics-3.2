#include "petscda.h"
#include "petscts.h"
#include "petscpc.h"
#include "petscsnes.h"

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

#include "TECIO.h"


struct UserCtx
{
    PetscInt IM, JM, KM;
    DA da, fda, fda2;
    DALocalInfo info;

    int IMax;
    int JMax;
    int KMax;

    int vertIII() const { return IMax * JMax * KMax; }
    int vertIdx(int k, int j, int i) const
    {
      return (k-info.zs) * IMax*JMax
           + (j-info.ys)*IMax
           + (i-info.xs);
    }

    int cellIII() const { return (IMax-1) * (JMax-1) * (KMax-1); }
    int cellIdx(int k, int j, int i) const
    {
      return (k-info.zs) * (IMax-1)*(JMax-1)
           + (j-info.ys) * (IMax-1)
           + (i-info.xs);
    }
    
    Vec Ucat;

    PetscInt binary_input;
    PetscInt conv_diff;
    PetscInt ti;
    PetscInt tis;
    PetscInt tsteps;
    PetscInt tie;
    PetscInt tempOpt;
    PetscInt ctemp;
    PetscInt rans;      // needs to be set
    PetscInt xyz_input; // needs to be set
    PetscInt block_number;

    // Character length
    PetscReal cl;

    // Constants
    int DIsDouble;

    std::string gridFilename;

    UserCtx() : rans(0)
              , xyz_input(0)
              , DIsDouble(0)
    {
    }
    
    // PetscInt xs, xe, ys, ye, zs, ze, mx, my, mz;
    // user->getInfo(xs, xe, ys, ye, zs, ze, mx, my, mz);
    void getInfo(PetscInt & xs, PetscInt & xe,
                 PetscInt & ys, PetscInt & ye,
                 PetscInt & zs, PetscInt & ze,
                 PetscInt & mx, PetscInt & my, PetscInt & mz)
    {
      xs = info.xs; xe = info.xs + info.xm;
      ys = info.ys; ye = info.ys + info.ym;
      zs = info.zs; ze = info.zs + info.zm;
      mx = info.mx; my = info.my; mz = info.mz;
    }
};


struct Cmpnts
{
    PetscScalar x, y, z;

    PetscScalar mag() const { return sqrt(x*x + y*y + z*z); }
};


bool FileExist(char const * filename)
{
  FILE *fd = fopen(filename, "r");
  if (!fd)
  {
    PetscPrintf(PETSC_COMM_WORLD, "\nCouldn't open %s file.\n\n", filename);
    return false;
  }
  fclose(fd);
  return true;
}


void FileReadError(char const * filename)
{
  PetscPrintf(PETSC_COMM_WORLD, "\nError reading %s file.\n\n", filename);
  exit(1);
}


PetscErrorCode ReadCoordinates(UserCtx * user)
{
  std::string filename = user->gridFilename;
  if (user->xyz_input)
    filename = std::string("xyz.dat");

  if (!FileExist(filename.c_str()))
    exit(1);

  FILE * fd = fopen(filename.c_str(), "r");

  PetscPrintf(PETSC_COMM_WORLD, "Begin Reading %s.\n", filename.c_str());


  // read number of blocks
  if (user->xyz_input)
    user->block_number = 1;

  else if (user->binary_input)
  {
    int cnt = fread(&user->block_number, sizeof(int), 1, fd);
    if (cnt != 1)
      FileReadError(filename.c_str());
    if (user->block_number != 1)
    {
      PetscPrintf(PETSC_COMM_WORLD, "This seems to be a text file.\n");
      exit(1);
    }
  }

  else
  {
    int cnt = fscanf(fd, "%i\n", &user->block_number);
    if (cnt != 1)
      FileReadError(filename.c_str());
    if (user->block_number != 1)
    {
      PetscPrintf(PETSC_COMM_WORLD, "This seems to be a binary file.\n");
      exit(1);
    }
  }

  // This program only handles one block.
  if (user->block_number != 1)
  {
    PetscPrintf(PETSC_COMM_WORLD,
                "This program only handles one block in grid.dat.\n"
                "This file has %d blocks!", user->block_number);
    exit(1);
  }


  // Read mesh size
  std::vector<double> X, Y,Z;

  if (user->xyz_input)
  {
    int cnt = fscanf(fd, "%i %i %i\n", &user->IM, &user->JM, &user->KM);
    if (cnt != 3)
      FileReadError(filename.c_str());

    X.resize(user->IM);
    Y.resize(user->JM);
    Z.resize(user->KM);

    double tmp;
    for (int i=0; i<user->IM; i++)
    {
      int cnt = fscanf(fd, "%le %le %le\n", &X[i], &tmp, &tmp);
      if (cnt != 3)
        FileReadError(filename.c_str());
    }
    for (int j=0; j<user->JM; j++)
    {
      int cnt = fscanf(fd, "%le %le %le\n", &tmp, &Y[j], &tmp);
      if (cnt != 3)
        FileReadError(filename.c_str());
    }
    for (int k=0; k<user->KM; k++)
    {
      int cnt = fscanf(fd, "%le %le %le\n", &tmp, &tmp, &Z[k]);
      if (cnt != 3)
        FileReadError(filename.c_str());
    }
  }
  else if (user->binary_input)
  {
    int cnt = fread(&user->IM, sizeof(int), 1, fd);
    if (cnt != 1)
      FileReadError(filename.c_str());

    cnt = fread(&user->JM, sizeof(int), 1, fd);
    if (cnt != 1)
      FileReadError(filename.c_str());

    cnt = fread(&user->KM, sizeof(int), 1, fd);
    if (cnt != 1)
      FileReadError(filename.c_str());
  }
  else
  {
    int cnt = fscanf(fd, "%i %i %i\n", &user->IM, &user->JM, &user->KM);
    if (cnt != 3)
      FileReadError(filename.c_str());
  }


  // Create grid
  DACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX,
             user->IM+1, user->JM+1, user->KM+1, 1, 1,
             PETSC_DECIDE, 1, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL, &user->da);

  if (user->rans) {
    DACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX,
               user->IM+1, user->JM+1, user->KM+1, 1,1,
               PETSC_DECIDE, 2, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL,
               &user->fda2);
  }
  DASetUniformCoordinates(user->da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  DAGetCoordinateDA(user->da, &user->fda);

  DAGetLocalInfo(user->da, &user->info);

  PetscInt xs, xe, ys, ye, zs, ze, mx, my, mz;
  user->getInfo(xs, xe, ys, ye, zs, ze, mx, my, mz);

  Vec Coor;
  Cmpnts ***coor;
  DAGetGhostedCoordinates(user->da, &Coor);
  DAVecGetArray(user->fda, Coor, &coor);


  PetscReal buffer;

  for (int k=0; k<user->KM; k++)
  for (int j=0; j<user->JM; j++)
  for (int i=0; i<user->IM; i++)
  {
    if (user->xyz_input)
    { }
    else if (user->binary_input)
    {
      int cnt = fread(&buffer, sizeof(double), 1, fd);
      if (cnt != 1)
        FileReadError(filename.c_str());
    }
    else
    {
      int cnt = fscanf(fd, "%le", &buffer);
      if (cnt != 1)
        FileReadError(filename.c_str());
    }

    if (k>=zs && k<=ze && j>=ys && j<ye && i>=xs && i<xe)
    {
      if(user->xyz_input)
        coor[k][j][i].x = X[i]/user->cl;
      else
        coor[k][j][i].x = buffer/user->cl;
    }
  }


  for (int k=0; k<user->KM; k++)
  for (int j=0; j<user->JM; j++)
  for (int i=0; i<user->IM; i++)
  {
    if (user->xyz_input)
    { }
    else if (user->binary_input)
    {
      int cnt = fread(&buffer, sizeof(double), 1, fd);
      if (cnt != 1)
        FileReadError(filename.c_str());
    }
    else
    {
      int cnt = fscanf(fd, "%le", &buffer);
      if (cnt != 1)
        FileReadError(filename.c_str());
    }

    if (k>=zs && k<=ze && j>=ys && j<ye && i>=xs && i<xe)
    {
      if(user->xyz_input)
        coor[k][j][i].y = Y[j]/user->cl;
      else
        coor[k][j][i].y = buffer/user->cl;
    }
  }


  for (int k=0; k<user->KM; k++)
  for (int j=0; j<user->JM; j++)
  for (int i=0; i<user->IM; i++) {
    if (user->xyz_input)
    { }
    else if (user->binary_input)
    {
      int cnt = fread(&buffer, sizeof(double), 1, fd);
      if (cnt != 1)
        FileReadError(filename.c_str());
    }
    else
    {
      int cnt = fscanf(fd, "%le", &buffer);
      if (cnt != 1)
        FileReadError(filename.c_str());
    }

    if (k>=zs && k<=ze && j>=ys && j<ye && i>=xs && i<xe)
    {
      if(user->xyz_input)
        coor[k][j][i].z = Z[k]/user->cl;
      else
        coor[k][j][i].z = buffer/user->cl;
    }
  }

  DAVecRestoreArray(user->fda, Coor, &coor);

  Vec gCoor;
  DAGetCoordinates(user->da, &gCoor);
  DALocalToGlobal(user->fda, Coor, INSERT_VALUES, gCoor);
  DAGlobalToLocalBegin(user->fda, gCoor, INSERT_VALUES, Coor);
  DAGlobalToLocalEnd(user->fda, gCoor, INSERT_VALUES, Coor);

  fclose(fd);

  PetscPrintf(PETSC_COMM_WORLD, "Finish Reading %s\n", filename.c_str());
  return(0);
}



PetscErrorCode WriteCoordinates(UserCtx * user)
{
  PetscInt xs, xe, ys, ye, zs, ze, mx, my, mz;
  user->getInfo(xs, xe, ys, ye, zs, ze, mx, my, mz);

  int ZoneType = 0;
  int ICellMax = 0;
  int JCellMax = 0;
  int KCellMax = 0;
  int IsBlock = 1;
  int NumFaceConnections = 0;
  int FaceNeighborMode = 0;
  /* 0 is cell-centered 1 is node centered */
  //             u  v  w  
  int LOC[40] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int ShareConnectivityFromZone = 0;

  /*************************/
  PetscPrintf(PETSC_COMM_WORLD, "mi=%d, mj=%d, mk=%d\n", mx, my, mz);
  PetscPrintf(PETSC_COMM_WORLD, "xs=%d, xe=%d\n", xs, xe);
  PetscPrintf(PETSC_COMM_WORLD, "ys=%d, ye=%d\n", ys, ye);
  PetscPrintf(PETSC_COMM_WORLD, "zs=%d, ze=%d\n", zs, ze);


  char filename[128];
  snprintf(filename, sizeof(filename), "Result%06d.plt", user->ti, 0);

  PetscPrintf(PETSC_COMM_WORLD, "\nGenerating file %s\n", filename);
        
  int Debug, VIsDouble;
  VIsDouble = 0;
  Debug = 0;

  std::string varTags("X Y Z U V W UU P Nv");

  if (user->conv_diff || user->ctemp)
    varTags += " C";
  
  if (user->tempOpt)
    varTags += " T";

  // PetscPrintf(PETSC_COMM_WORLD, "tecplot var tags: %s\n", varTags.c_str());

  int I = TECINI100(const_cast<char *>("Flow"), const_cast<char *>(varTags.c_str()),
                    filename, const_cast<char *>("."), &Debug, &VIsDouble);


  user->IMax = mx - 1;
  user->JMax = my - 1;
  user->KMax = mz - 1;

  I = TECZNE100(const_cast<char *>("Block 1"),
                &ZoneType,      /* Ordered zone */
                &user->IMax,
                &user->JMax,
                &user->KMax,
                &ICellMax,
                &JCellMax,
                &KCellMax,
                &IsBlock,       /* ISBLOCK  BLOCK format */
                &NumFaceConnections,
                &FaceNeighborMode,
                LOC,
                NULL,
                &ShareConnectivityFromZone); /* No connectivity sharing */

  Vec Coor;
  Cmpnts ***coor;
  DAGetCoordinates(user->da, &Coor);
  DAVecGetArray(user->fda, Coor, &coor);

  int III = user->vertIII();
  std::unique_ptr<float[]> x(new float[III]);

  // PetscPrintf(PETSC_COMM_WORLD, "Coord: number of tecplot data values: %d\n", 3 * III);

  for (int k=zs; k<ze-1; k++)
  for (int j=ys; j<ye-1; j++)
  for (int i=xs; i<xe-1; i++)
    x[user->vertIdx(k, j, i)] = coor[k][j][i].x;
  I = TECDAT100(&III, &x[0], &user->DIsDouble);

  for (int k=zs; k<ze-1; k++)
  for (int j=ys; j<ye-1; j++)
  for (int i=xs; i<xe-1; i++)
    x[user->vertIdx(k, j, i)] = coor[k][j][i].y;
  I = TECDAT100(&III, &x[0], &user->DIsDouble);

  for (int k=zs; k<ze-1; k++)
  for (int j=ys; j<ye-1; j++)
  for (int i=xs; i<xe-1; i++)
    x[user->vertIdx(k, j, i)] = coor[k][j][i].z;
  I = TECDAT100(&III, &x[0], &user->DIsDouble);

  DAVecRestoreArray(user->fda, Coor, &coor);
}


PetscErrorCode WriteVelocity(UserCtx * user)
{
  // Load ucat
  char filename[128];
  snprintf(filename, sizeof(filename), "ufield%06d_%1d.dat", user->ti, 0);
  if (!FileExist(filename))
    exit(1);

  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
  DACreateGlobalVector(user->fda, &user->Ucat);
  VecLoadIntoVector(viewer, user->Ucat);
  PetscViewerDestroy(viewer);

  Cmpnts ***ucat;
  DAVecGetArray(user->fda, user->Ucat, &ucat);

  PetscInt xs, xe, ys, ye, zs, ze, mx, my, mz;
  user->getInfo(xs, xe, ys, ye, zs, ze, mx, my, mz);

  int I, III = user->cellIII();
  std::unique_ptr<float[]> x(new float[III]);

  // PetscPrintf(PETSC_COMM_WORLD, "Velocity: number of tecplot data values: %d\n", 4 * III);

  for (int k=zs; k<ze-2; k++)
  for (int j=ys; j<ye-2; j++)
  for (int i=xs; i<xe-2; i++)
    x[user->cellIdx(k, j, i)] = ucat[k+1][j+1][i+1].x;
  I = TECDAT100(&III, &x[0], &user->DIsDouble);

  for (int k=zs; k<ze-2; k++)
  for (int j=ys; j<ye-2; j++)
  for (int i=xs; i<xe-2; i++)
    x[user->cellIdx(k, j, i)] = ucat[k+1][j+1][i+1].y;
  I = TECDAT100(&III, &x[0], &user->DIsDouble);

  for (int k=zs; k<ze-2; k++)
  for (int j=ys; j<ye-2; j++)
  for (int i=xs; i<xe-2; i++)
    x[user->cellIdx(k, j, i)] = ucat[k+1][j+1][i+1].z;
  I = TECDAT100(&III, &x[0], &user->DIsDouble);
			
  for (int k=zs; k<ze-2; k++)
  for (int j=ys; j<ye-2; j++)
  for (int i=xs; i<xe-2; i++)
    x[user->cellIdx(k, j, i)] = ucat[k+1][j+1][i+1].mag();
  I = TECDAT100(&III, &x[0], &user->DIsDouble);

  DAVecRestoreArray(user->fda, user->Ucat, &ucat);
  VecDestroy(user->Ucat);
}


void WriteFieldCell(UserCtx * user, std::string const & _filename)
{
  Vec Field;
  DACreateGlobalVector(user->da, &Field);

  char filename[128];
  snprintf(filename, sizeof(filename), _filename.c_str(), user->ti, 0);
  if (!FileExist(filename))
    exit(1);

  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
  VecLoadIntoVector(viewer, Field);
  PetscViewerDestroy(viewer);

  PetscReal ***field;
  DAVecGetArray(user->da, Field, &field);

  PetscInt xs, xe, ys, ye, zs, ze, mx, my, mz;
  user->getInfo(xs, xe, ys, ye, zs, ze, mx, my, mz);

  int I, III = user->cellIII();
  std::unique_ptr<float[]> x(new float[III]);

  // PetscPrintf(PETSC_COMM_WORLD, "%s: number of tecplot data values: %d\n", filename, III);

  for (int k=zs; k<ze-2; k++)
  for (int j=ys; j<ye-2; j++)
  for (int i=xs; i<xe-2; i++)
    x[user->cellIdx(k, j, i)] = field[k+1][j+1][i+1];

  I = TECDAT100(&III, &x[0], &user->DIsDouble);
  
  DAVecRestoreArray(user->da, Field, &field);
  VecDestroy(Field);
}


void TECClose()
{
  int I = TECEND100();
}


int main(int argc, char **argv)
{
  FILE *fd = fopen("control.dat", "r");
  if (!fd)
  {
    printf("\nCouldn't open %s file.\n\n", "control.dat");
    exit(1);
  }
  fclose(fd);

  PetscInitialize(&argc, &argv, "control.dat",
                  "Convert VFS files to tecplot - small footprint.");


  std::unique_ptr<UserCtx> user(new UserCtx);

  PetscTruth set;
  {
    user->gridFilename = std::string("grid.dat");
    char filename[256];
    PetscOptionsGetString(PETSC_NULL,"-grid", filename, 256, &set);
    if (set)
      user->gridFilename = std::string(filename);
  }

  user->binary_input = 0;
  PetscOptionsGetInt(PETSC_NULL, "-binary", &user->binary_input, PETSC_NULL);

  user->conv_diff = 0;
  PetscOptionsGetInt(PETSC_NULL, "-conv_diff", &user->conv_diff, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-tis", &user->tis, &set);
  if (!set)
  {
    PetscPrintf(PETSC_COMM_WORLD,
                "Program needs a starting number (-tis option)!\n");
    exit(1);
  }

  PetscOptionsGetInt(PETSC_NULL, "-tie", &user->tie, &set);
  if (!set)
    user->tie = user->tis;

  PetscOptionsGetInt(PETSC_NULL, "-ts", &user->tsteps, &set);
  if (!set)
    user->tsteps = 50;

  user->tempOpt = 0;
  PetscOptionsGetInt(PETSC_NULL, "-temp", &user->tempOpt, &set);

  user->ctemp = 0;
  PetscOptionsGetInt(PETSC_NULL, "-conv_temp", &user->ctemp, &set);
  if (set)
    user->conv_diff = 0;

  // PetscReal  cl = 1.;
  user->cl = 1.0;
  PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &user->cl, &set);
  if (!set)
    PetscOptionsGetReal(PETSC_NULL, "-cl", &user->cl, PETSC_NULL);

  ReadCoordinates(user.get());

  for (user->ti=user->tis; user->ti<=user->tie; user->ti+=user->tsteps)
  {
    PetscPrintf(PETSC_COMM_WORLD, "\n\nProcessing timestep %d\n", user->ti);

    WriteCoordinates(user.get());
    WriteVelocity(user.get());

    WriteFieldCell(user.get(), "pfield%06d_%1d.dat");
    WriteFieldCell(user.get(), "nvfield%06d_%1d.dat");

    if (user->conv_diff || user->ctemp)
      WriteFieldCell(user.get(), "cfield%06d_%1d.dat");

    if (user->tempOpt)
      WriteFieldCell(user.get(), "tfield%06d_%1d.dat");
    
    TECClose();
  }

  PetscFinalize();
}
