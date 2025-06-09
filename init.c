#include "variables.h"
/*
extern int block_number, inletprofile;
extern PetscReal L_dim;
extern int i_proc, j_proc, k_proc;
extern int binary_input, xyz_input;
extern int averaging;// seokkoo
*/
PetscErrorCode FormInitialize(UserCtx *user)
{
  DM		fda = user->fda;
  Vec		Ucont = user->Ucont;
  Cmpnts	***ucont;

  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;

  int	i, j, k;

  DMDAVecGetArray(fda, Ucont, &ucont);

  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	ucont[k][j][i].x = 0;
	ucont[k][j][i].y = 0;
	ucont[k][j][i].z = 0;
      }
    }
  }

  DMDAVecRestoreArray(fda, Ucont, &ucont);
  VecCopy(Ucont, user->Ucat);
  VecCopy(Ucont, user->Bcs.Ubcs);

  DMDAVecGetArray(fda, user->Bcs.Ubcs, &ucont);

  VecSet(user->Phi, 0.);
  VecSet(user->DUold, 0.);
  VecSet(user->lNvert, 0.);

  /*  if (ze==mz) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	k=0;
	ucont[k][j][i].z = Flux_in;
	k=ze-1;
	ucont[k][j][i].z = Flux_in;
      }
    }
    }*/
  DMDAVecRestoreArray(fda, user->Bcs.Ubcs, &ucont);
  user->ren = 3000.;
/*   user->dt = 0.01; */
/*   user->st = 1.; */
  user->dt = 0.04;
  
  user->cfl=0.01;
  user->vnn=0.01;
  user->Ri = 0.000000001; 
  if (density_current)
    PetscOptionsGetReal(PETSC_NULL, "-richardson", &user->Ri, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-ren", &user->ren, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-dt", &user->dt, PETSC_NULL);
  dt_inflow = user->dt;
  PetscOptionsGetReal(PETSC_NULL, "-dt_inflow", &dt_inflow, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-cfl", &user->cfl, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-vnn", &user->vnn, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-Pr", &user->Pr, PETSC_NULL); 
  PetscOptionsGetReal(PETSC_NULL, "-Ri_x", &user->Ri_x, PETSC_NULL); 
  PetscOptionsGetReal(PETSC_NULL, "-Ri_y", &user->Ri_y, PETSC_NULL); 
  PetscOptionsGetReal(PETSC_NULL, "-Ri_z", &user->Ri_z, PETSC_NULL); 


  user->st = 1.;//0.038406145;
  
  PetscPrintf(PETSC_COMM_WORLD, "Re %le St %le\n",user->ren,user->st);

  return(0);
}

PetscErrorCode MGDACreate(UserMG *usermg, int bi)
{
  MGCtx *mgctx = usermg->mgctx;

  UserCtx *user, *user_high;


  int l;//, bi;
/*   PetscMalloc(usermg->mglevels*sizeof(int), &IM); */
/*   PetscMalloc(usermg->mglevels*sizeof(int), &JM); */
/*   PetscMalloc(usermg->mglevels*sizeof(int), &KM); */

  for (l=usermg->mglevels-2; l>=0; l--) {
    user = mgctx[l].user;
    user_high = mgctx[l+1].user;
    //for (bi = 0; bi<block_number; bi++) {
      
      if (usermg->isc) {	 
	user[bi].IM = user_high[bi].IM;
      }
      else {
	user[bi].IM = (user_high[bi].IM + 1) / 2;
      }
      if (usermg->jsc) {
	user[bi].JM = user_high[bi].JM;
      }
      else {
	user[bi].JM = (user_high[bi].JM + 1) / 2;
      }

      if (usermg->ksc) {
	user[bi].KM = user_high[bi].KM;
      }
      else {
	user[bi].KM = (user_high[bi].KM + 1) / 2;
      }

      if (user[bi].IM*(2-usermg->isc)-(user_high[bi].IM+1-usermg->isc) +
	  user[bi].JM*(2-usermg->jsc)-(user_high[bi].JM+1-usermg->jsc) +
	  user[bi].KM*(2-usermg->ksc)-(user_high[bi].KM+1-usermg->ksc)) {
	PetscPrintf(PETSC_COMM_WORLD, "Grid at level %d can't be further restricted!", l);
	// return 1;
      }
      //}
  }

  l = 0;
  user = mgctx[l].user;
  //for (bi=0; bi<block_number; bi++) {
	// m,n,p : corresponding number of processors in each dimension ; seokkoo
	int total_rank;
	MPI_Comm_size(PETSC_COMM_WORLD, &total_rank);
	PetscInt m, n, p, s;
	//DAPeriodicType wrap;
	DMDABoundaryType bx=DMDA_BOUNDARY_GHOSTED, by=DMDA_BOUNDARY_GHOSTED, bz=DMDA_BOUNDARY_GHOSTED;
	bx=DMDA_BOUNDARY_NONE, by=DMDA_BOUNDARY_NONE, bz=DMDA_BOUNDARY_NONE;
  
	m = n = p = PETSC_DECIDE;
  
	m=i_proc, n=j_proc, p=k_proc;
	
	if(i_periodic) m=1;
	if(j_periodic) n=1;
	if(k_periodic) p=1;
  	
	if(ii_periodic || jj_periodic || kk_periodic || levelset) s=3;
	else s=3;
  
	if(ii_periodic) bx = DMDA_BOUNDARY_PERIODIC;
	if(jj_periodic) by = DMDA_BOUNDARY_PERIODIC;
	if(kk_periodic) bz = DMDA_BOUNDARY_PERIODIC;
	
	
	DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX,
	       user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, m, n,
	       p, 1, s, PETSC_NULL, PETSC_NULL, PETSC_NULL,
	       &(user[bi].da));
	
	if(rans)
	DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX,
	       user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, m, n,
	       p, 2, s, PETSC_NULL, PETSC_NULL, PETSC_NULL,
	       &(user[bi].fda2));
	       
	bi=0;
	DMDAGetInfo(user[bi].da, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &m, &n, &p, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  //DMDAGetInfo(user[bi].da, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &m, &n, &p, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
	PetscPrintf(PETSC_COMM_WORLD, "**DM Distribution: %i %i %i\n", m, n, p); //seokkoo
  /*
    DMDACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DMDA_STENCIL_BOX,
	       user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, PETSC_DECIDE, PETSC_DECIDE,
	       PETSC_DECIDE, 1, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL,
	       &(user[bi].da));    
  */
    user[bi].aotopetsc = PETSC_FALSE;
    DMDASetUniformCoordinates(user[bi].da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    DMDAGetCoordinateDA(user[bi].da, &(user[bi].fda));
    DMDAGetLocalInfo(user[bi].da, &(user[bi].info));
    

    // }
    
  for (l=1; l<usermg->mglevels; l++) {
    user = mgctx[l].user;
    //for (bi=0; bi<block_number; bi++) {
      //DMDAGetInfo(mgctx[0].user[bi].da, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &m, &n, &p, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
      DMDAGetInfo(mgctx[0].user[bi].da, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &m, &n, &p, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
      PetscPrintf(PETSC_COMM_WORLD, "DM Distribution: %i %i %i\n", m, n, p);
	
      DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX, user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, m, n, p, 1, s, PETSC_NULL, PETSC_NULL, PETSC_NULL, &(user[bi].da));
      user[bi].aotopetsc = PETSC_FALSE;
      DMDASetUniformCoordinates(user[bi].da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
      DMDAGetCoordinateDA(user[bi].da, &(user[bi].fda));
      DMDAGetLocalInfo(user[bi].da, &(user[bi].info));

      //}
  }
  return 0;
}



PetscErrorCode MG_Initial(UserMG *usermg, IBMNodes *ibm)
{
	MGCtx *mgctx;

	int level;

	int rank;
	int IM, JM, KM;

	int i, j, k;
	int bi;

	PetscErrorCode ierr;

	Vec Coor, gCoor;
	Cmpnts ***coor;
	  
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscReal cl = 1.;

	PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &cl, PETSC_NULL);

  /* How many MG levels, the default is 3 */
	
	usermg->mglevels = 1;	// seokkoo
	PetscOptionsGetInt(PETSC_NULL, "-mg_level", &usermg->mglevels, PETSC_NULL);
  
	if(poisson!=-1) usermg->mglevels = 1;	// seokkoo
	
	usermg->ksc = PETSC_FALSE;
	usermg->jsc = PETSC_FALSE;
	usermg->isc = PETSC_FALSE;

	PetscOptionsGetBool(PETSC_NULL, "-mg_k_semi", &usermg->ksc, PETSC_NULL);
	PetscOptionsGetBool(PETSC_NULL, "-mg_j_semi", &usermg->jsc, PETSC_NULL);
	PetscOptionsGetBool(PETSC_NULL, "-mg_i_semi", &usermg->isc, PETSC_NULL);
	PetscMalloc(usermg->mglevels*sizeof(MGCtx), &(usermg->mgctx));
	mgctx = usermg->mgctx;
  
	FILE *fd;
  
  /* Read in number of blocks and allocate memory for UserCtx */
	
	char str[256];
	
	if(xyz_input) sprintf(str, "%s/%s", path, "xyz.dat");
	else sprintf(str, "%s/%s", path, gridfile);
	
	fd = fopen(str, "r");
	if(fd==NULL) printf("Cannot open %s !\n", str),exit(1);
	
	if(xyz_input) {block_number=1;}
	else if(binary_input) fread(&block_number, sizeof(int), 1, fd);
	else fscanf(fd, "%i\n", &block_number);
	
	for (level=0; level<usermg->mglevels; level++) {
		PetscMalloc(block_number*sizeof(UserCtx), &mgctx[level].user);
		for (bi=0; bi<block_number; bi++) {
			mgctx[level].user[bi].ibm = ibm;
			mgctx[level].user[bi].isc = &usermg->isc;
			mgctx[level].user[bi].jsc = &usermg->jsc;
			mgctx[level].user[bi].ksc = &usermg->ksc;
		}
	}

  /* Read from grid.dat the number of grid points along I, J, K directions
     and the detail coordinates of grid nodes */
	level = usermg->mglevels-1;
	UserCtx *user;

	user = mgctx[level].user;
	/*
	if (inletprofile==3) {
		PetscPrintf(PETSC_COMM_WORLD, "READ INFLOW WAVE FORM!!!! mg_init\n");
		for (bi=0; bi<block_number; bi++) {
			InflowWaveFormRead(&user[bi]);
		}
		PetscPrintf(PETSC_COMM_WORLD, "READ INFLOW WAVE FORM!!!! mg_init\n");
	}
	*/
	


        // <WRO> read temperature and buoyancy parameters here.
  
        user->temp = 0;
        PetscOptionsGetInt(PETSC_NULL, "-temp", &user->temp, PETSC_NULL);
  
        PetscBool set;
        user->ctemp = 0;
        PetscOptionsGetInt(PETSC_NULL, "-conv_temp", &user->ctemp, &set);
        if (set)
          conv_diff = user->ctemp;

        // Initial condition for temperature.
        user->temp_IC = 0.0;
        PetscOptionsGetReal(PETSC_NULL, "-temp_ic", &user->temp_IC, PETSC_NULL);

        // This file specifies the buoyancy BC. Specifically, it identifies
        // the cells that have a Dirichlet BC set to -buoy_temp.
        { // put in {} to reduce lifetime of buoyFilename.
          user->buoyFilenameSet = PETSC_FALSE;


          new (&user->buoyFilename) std::string("");

          user->buoyFilename = std::string("");
          char buoyFilename[256] = "";
          PetscOptionsGetString(PETSC_NULL,"-buoy_bc_filename", buoyFilename, 256,
                                &user->buoyFilenameSet);
          if (user->buoyFilenameSet)
            user->buoyFilename = std::string(buoyFilename);
        }

        // Run the buoyancy for the mouth/nose case
        user->buoyMouthNose = 0;
        PetscOptionsGetInt(PETSC_NULL, "-buoy_mouth_nose", &user->buoyMouthNose, PETSC_NULL);

        // Run the test case for buoyancy
        user->buoyTest = 0;
        PetscOptionsGetInt(PETSC_NULL, "-buoy_test", &user->buoyTest, PETSC_NULL);

        // Reference temperature for buoyancy calculation (degrees C)
        // This is the default temperature for the whole domain.
        user->buoyTempRef = 0.0;
        user->buoyBeta = 0.0;
        PetscOptionsGetReal(PETSC_NULL, "-buoy_temp_ref", &user->buoyTempRef, &set);
        // If the reference temperature is set, calculate the thermal expansion
        // coefficient (beta): the inverse of temp ref in Kelvin.
        if (set)
          user->buoyBeta = 1.0 / (user->buoyTempRef + 273.15);
  
        // For -mouth_nose_buoy and -test_buoy,
        // the temp of hot or cold areas (degrees C)
        // This is the temperature of the body causing the buoyancy flow.
        user->buoyTemp = 0.0;
        PetscOptionsGetReal(PETSC_NULL, "-buoy_temp", &user->buoyTemp, PETSC_NULL);

        // If needed, read the temperature BC file.
        if (user->temp && user->buoyFilenameSet)
          readBuoyBC(user);


	for (bi=0; bi<block_number; bi++) {
		
		std::vector<double> X, Y,Z;
		double tmp;
		
		if(xyz_input) {
			fscanf(fd, "%i %i %i\n", &(user[bi].IM), &(user[bi].JM), &(user[bi].KM));
			X.resize(user[bi].IM);
			Y.resize(user[bi].JM);
			Z.resize(user[bi].KM);
			
			for (i=0; i<user[bi].IM; i++) fscanf(fd, "%le %le %le\n", &X[i], &tmp, &tmp);
			for (j=0; j<user[bi].JM; j++) fscanf(fd, "%le %le %le\n", &tmp, &Y[j], &tmp);
			for (k=0; k<user[bi].KM; k++) fscanf(fd, "%le %le %le\n", &tmp, &tmp, &Z[k]);
		}
		else if(binary_input) {
			fread(&(user[bi].IM), sizeof(int), 1, fd);
			fread(&(user[bi].JM), sizeof(int), 1, fd);
			fread(&(user[bi].KM), sizeof(int), 1, fd);
		}
		else fscanf(fd, "%i %i %i\n", &(user[bi].IM), &(user[bi].JM), &(user[bi].KM));
		
		IM = user[bi].IM;
		JM = user[bi].JM;
		KM = user[bi].KM;
		
		PetscPrintf(PETSC_COMM_WORLD, "Reading %s %le, %dx%dx%d\n", gridfile, L_dim, IM, JM, KM);

		MGDACreate(usermg, bi);
		MPI_Barrier(PETSC_COMM_WORLD);
		PetscPrintf(PETSC_COMM_WORLD, "Created DM\n");

		DMDALocalInfo	info = user[bi].info;
		int	xs = info.xs, xe = info.xs + info.xm;
		int  	ys = info.ys, ye = info.ys + info.ym;
		int	zs = info.zs, ze = info.zs + info.zm;

		DMDAGetGhostedCoordinates(user[bi].da, &Coor);
		DMDAVecGetArray(user[bi].fda, Coor, &coor);
		
		double buffer;
		
		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
				
			if(xyz_input) {}
			else if(binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);
				
			if( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if(xyz_input) coor[k][j][i].x = X[i]/cl*L_dim;
				else coor[k][j][i].x = buffer/cl*L_dim;
			}
		}
			
		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if(xyz_input) {}
			else if(binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);
				
			if( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if(xyz_input) coor[k][j][i].y = Y[j]/cl*L_dim;
				else coor[k][j][i].y = buffer/cl*L_dim;
			}
		}
	
		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if(xyz_input) {}
			else if(binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);
				
			if( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if(xyz_input) coor[k][j][i].z = Z[k]/cl*L_dim;
				else coor[k][j][i].z = buffer/cl*L_dim;
			}
		}
	      /*
			for (k=zs; k<ze; k++)
			for (j=ys; j<ye; j++)
			for (i=xs; i<xe; i++) {
				if (k<KM && j<JM && i<IM) {
					coor[k][j][i].x = *(gc + (k * (IM*JM) + j * IM + i) * 3  )/cl*L_dim;
					coor[k][j][i].y = *(gc + (k * (IM*JM) + j * IM + i) * 3+1)/cl*L_dim;
					coor[k][j][i].z = *(gc + (k * (IM*JM) + j * IM + i) * 3+2)/cl*L_dim;
				}
			}
		*/
    
		DMDAVecRestoreArray(user[bi].fda, Coor, &coor);
		DMDAGetCoordinates(user[bi].da, &gCoor);
		
		DMLocalToGlobalBegin(user[bi].fda, Coor, INSERT_VALUES, gCoor);
		DMLocalToGlobalEnd(user[bi].fda, Coor, INSERT_VALUES, gCoor);
		
		DMGlobalToLocalBegin(user[bi].fda, gCoor, INSERT_VALUES, Coor);
		DMGlobalToLocalEnd(user[bi].fda, gCoor, INSERT_VALUES, Coor);
	}
	
	
	fclose(fd);
	MPI_Barrier(PETSC_COMM_WORLD);
	
  

	if(poisson==-1) {
		// removed by seokkoo
			UserCtx *user_high;
			Vec Coor, gCoor, Coor_high;
			Cmpnts ***coor_high;

		  for (level=usermg->mglevels-2; level>-1; level--) {
		    user = mgctx[level].user;
		    user_high = mgctx[level+1].user;
		    for (bi = 0; bi<block_number; bi++) {
		      PetscPrintf(PETSC_COMM_WORLD, "aaaa %i\n", level);
		      DMDALocalInfo	info = user[bi].info;
		      int	xs = info.xs, xe = info.xs + info.xm;
		      int  ys = info.ys, ye = info.ys + info.ym;
		      int	zs = info.zs, ze = info.zs + info.zm;
		      int	mx = info.mx, my = info.my, mz = info.mz;

		      DMDAGetGhostedCoordinates(user_high[bi].da, &Coor_high);
		      DMDAGetCoordinates(user[bi].da, &Coor);

		      DMDAVecGetArray(user_high[bi].fda, Coor_high, &coor_high);
		      DMDAVecGetArray(user[bi].fda, Coor, &coor);

		      if (xe==mx) xe--;
		      if (ye==my) ye--;
		      if (ze==mz) ze--;
		      
		      for (k=zs; k<ze; k++) {
			for (j=ys; j<ye; j++) {
			  for (i=xs; i<xe; i++) {

				int ih, jh, kh;
			    GridRestriction(i, j, k, &ih, &jh, &kh, &user[bi]);
				
			    coor[k][j][i].x = coor_high[kh][jh][ih].x;
			    coor[k][j][i].y = coor_high[kh][jh][ih].y;
			    coor[k][j][i].z = coor_high[kh][jh][ih].z;
			  }
			}
		      }
		      
		      DMDAVecRestoreArray(user[bi].fda, Coor, &coor);
		      DMDAVecRestoreArray(user_high[bi].fda, Coor_high, &coor_high);

		      DMDAGetGhostedCoordinates(user[bi].da, &gCoor);

		      DMGlobalToLocalBegin(user[bi].fda, Coor, INSERT_VALUES, gCoor);
		      DMGlobalToLocalEnd(user[bi].fda, Coor, INSERT_VALUES, gCoor);


		    }
		  }
	  }
	

	level = usermg->mglevels-1;
	user = mgctx[level].user;
	if (!rank) {
		if (block_number>1) {
			char str[256];
			sprintf(str, "%s/interface.dat", path);
			fd = fopen(str, "r");
			for (bi=0; bi<block_number; bi++) {
				fscanf(fd, "%i\n", &(user[bi].itfcptsnumber));
				MPI_Bcast(&(user[bi].itfcptsnumber), 1, MPI_INT, 0, PETSC_COMM_WORLD);
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfcI));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfcJ));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfcK));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostI));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostJ));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostK));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostB));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(PetscReal),&(user[bi].itfchostx));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(PetscReal),&(user[bi].itfchosty));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(PetscReal),&(user[bi].itfchostz));
				
				for (i=0; i<user[bi].itfcptsnumber; i++) {
					fscanf(fd, "%i %i %i\n", &(user[bi].itfcI[i]), &(user[bi].itfcJ[i]),&(user[bi].itfcK[i]));
					fscanf(fd, "%i %i %i %i\n", &(user[bi].itfchostI[i]),&(user[bi].itfchostJ[i]), &(user[bi].itfchostK[i]),&(user[bi].itfchostB[i]));
					fscanf(fd, "%le %le %le\n", &(user[bi].itfchostx[i]),&(user[bi].itfchosty[i]), &(user[bi].itfchostz[i]));
				}
				MPI_Bcast(user[bi].itfcI, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfcJ, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfcK, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
      
				MPI_Bcast(user[bi].itfchostI, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostJ, user[bi].itfcptsnumber, MPI_INT, 0,  PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostK, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostB, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostx, user[bi].itfcptsnumber, MPIU_REAL, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchosty, user[bi].itfcptsnumber, MPIU_REAL, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostz, user[bi].itfcptsnumber, MPIU_REAL, 0, PETSC_COMM_WORLD);
			}
			fclose(fd);
		}
    
		/* Read in bcs.dat for boundary conditions at 6 boundary surfaces 
		First put the data onto the finest level and restrict to the coarser
		levels */
		char str[256];
		sprintf(str, "%s/bcs.dat", path);
		fd = fopen(str, "r");
		if(!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(1);

		for (bi=0; bi<block_number; bi++) {
                  fscanf(fd, "%i %i %i %i %i %i\n",
                         &(user[bi].bctype[0]), &(user[bi].bctype[1]),
                         &(user[bi].bctype[2]), &(user[bi].bctype[3]),
                         &(user[bi].bctype[4]), &(user[bi].bctype[5]));
			MPI_Bcast(&(user[bi].bctype[0]), 6, MPI_INT, 0, PETSC_COMM_WORLD);
		}

		// begin add (xiaolei)
                if (user->temp) {
                        PetscPrintf(PETSC_COMM_WORLD, "Reading BCs for Temperature !\n");
                        for (bi=0; bi<block_number; bi++) {
                          fscanf(fd, "%i %i %i %i %i %i\n",
                                 &(user[bi].bctype_tmprt[0]), &(user[bi].bctype_tmprt[1]),
                                 &(user[bi].bctype_tmprt[2]), &(user[bi].bctype_tmprt[3]),
                                 &(user[bi].bctype_tmprt[4]), &(user[bi].bctype_tmprt[5]));
                                MPI_Bcast(&(user[bi].bctype_tmprt[0]), 6, MPI_INT, 0, PETSC_COMM_WORLD);
                        }

                        for (bi=0; bi<block_number; bi++) {
                                fscanf(fd, "%le %le %le %le %le %le\n",
                                       &(user[bi].tmprts[0]), &(user[bi].tmprts[1]),
                                       &(user[bi].tmprts[2]), &(user[bi].tmprts[3]),
                                       &(user[bi].tmprts[4]),&(user[bi].tmprts[5]));
                                MPI_Bcast(&(user[bi].tmprts[0]), 6, MPIU_REAL, 0, PETSC_COMM_WORLD);

                        }
                }

		// end add (xiaolei)


		fclose(fd);
		
		for(int ibi=0; ibi<NumberOfBodies; ibi++) ib_bctype[ibi]=0;
		if(immersed) {
			sprintf(str, "%s/ib_bcs.dat", path);
			fd = fopen(str, "r");
			if(fd) {
				for(int ibi=0; ibi<NumberOfBodies; ibi++) {
					if(!feof(fd)) fscanf(fd, "%i", &ib_bctype[ibi]);
				}
				fclose(fd);
				printf("Read %s\n", str);
			}
			MPI_Bcast(&(ib_bctype[0]), NumberOfBodies, MPI_INT, 0, PETSC_COMM_WORLD);
		}
	}  
	else {
		if (block_number>1) {  
			for (bi=0; bi<block_number; bi++) {
				MPI_Bcast(&(user[bi].itfcptsnumber), 1, MPI_INT, 0, PETSC_COMM_WORLD);
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfcI));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfcJ));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfcK));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostI));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostJ));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostK));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(int),&(user[bi].itfchostB));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(PetscReal),&(user[bi].itfchostx));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(PetscReal),&(user[bi].itfchosty));
				PetscMalloc((user[bi].itfcptsnumber)*sizeof(PetscReal),&(user[bi].itfchostz));
				
				MPI_Bcast(user[bi].itfcI, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfcJ, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfcK, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostI, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostJ, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostK, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostB, user[bi].itfcptsnumber, MPI_INT, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostx, user[bi].itfcptsnumber, MPIU_REAL, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchosty, user[bi].itfcptsnumber, MPIU_REAL, 0, PETSC_COMM_WORLD);
				MPI_Bcast(user[bi].itfchostz, user[bi].itfcptsnumber, MPIU_REAL, 0, PETSC_COMM_WORLD);
			}
		}
		
		for (bi=0; bi<block_number; bi++) {
			MPI_Bcast(&(user[bi].bctype[0]), 6, MPI_INT, 0, PETSC_COMM_WORLD);
		}
		
		if(immersed) MPI_Bcast(&(ib_bctype[0]), NumberOfBodies, MPI_INT, 0, PETSC_COMM_WORLD);


		// begin add (xiaolei)
                if (user->temp) {
                        for (bi=0; bi<block_number; bi++) {
                                MPI_Bcast(&(user[bi].bctype_tmprt[0]), 6, MPI_INT, 0, PETSC_COMM_WORLD);
                        }

                        for (bi=0; bi<block_number; bi++) {
                                MPI_Bcast(&(user[bi].tmprts[0]), 6, MPIU_REAL, 0, PETSC_COMM_WORLD);
                        }
                }

		// end add (xiaolei)


	}
	
	for (level = usermg->mglevels-2; level>=0; level--) {
		user = mgctx[level].user;
		for (bi=0; bi<block_number; bi++) {
			user[bi].bctype[0] = mgctx[level+1].user[bi].bctype[0];
			user[bi].bctype[1] = mgctx[level+1].user[bi].bctype[1];
			user[bi].bctype[2] = mgctx[level+1].user[bi].bctype[2];
			user[bi].bctype[3] = mgctx[level+1].user[bi].bctype[3];
			user[bi].bctype[4] = mgctx[level+1].user[bi].bctype[4];
			user[bi].bctype[5] = mgctx[level+1].user[bi].bctype[5];
		}
	}
	
	for (level=usermg->mglevels-1; level>=0; level--) {
		user = mgctx[level].user;
		for (bi=0; bi<block_number; bi++) {
			user[bi].thislevel = level;
			user[bi]._this = bi;
			user[bi].mglevels = usermg->mglevels;
			if (level > 0) {
				user[bi].da_c = &mgctx[level-1].user[bi].da;
				user[bi].lNvert_c = &mgctx[level-1].user[bi].lNvert;
				user[bi].user_c = &mgctx[level-1].user[bi];
			}
			if (level < usermg->mglevels-1) {
				user[bi].da_f = &mgctx[level+1].user[bi].da;
				user[bi].user_f = &mgctx[level+1].user[bi];
			}
			//  PetscPrintf(PETSC_COMM_WORLD, "Number %d", ibm.n_elmt);
			//  PetscBarrier((PetscObject)user.da);
			ierr = DMCreateGlobalVector(user[bi].fda, &(user[bi].Csi));
			ierr = VecDuplicate(user[bi].Csi, &(user[bi].Eta));
			ierr = VecDuplicate(user[bi].Csi, &(user[bi].Zet));

			VecDuplicate(user[bi].Csi, &(user[bi].ICsi));
			VecDuplicate(user[bi].Csi, &(user[bi].IEta));
			VecDuplicate(user[bi].Csi, &(user[bi].IZet));
			VecDuplicate(user[bi].Csi, &(user[bi].JCsi));
			VecDuplicate(user[bi].Csi, &(user[bi].JEta));
			VecDuplicate(user[bi].Csi, &(user[bi].JZet));
			VecDuplicate(user[bi].Csi, &(user[bi].KCsi));
			VecDuplicate(user[bi].Csi, &(user[bi].KEta));
			VecDuplicate(user[bi].Csi, &(user[bi].KZet));
			VecDuplicate(user[bi].Csi, &(user[bi].Cent));

			// begin add (xiaolei)
                        // added by xyang 12-7-2010
                        if (rotor_model || IB_delta)
                          VecDuplicate(user[bi].Csi, &(user[bi].F_eul));

                        // added by xyang 
                        if (user->temp)
                          VecDuplicate(user[bi].Csi, &(user[bi].gFTmprt));

                        // added by xyang 
                        if (conv_diff)
                          VecDuplicate(user[bi].Csi, &(user[bi].FCurrent));

                        if (Force_wm && (imin_wm !=0 || imax_wm !=0 || jmin_wm != 0 || jmax_wm !=0 || (IB_wm != 0 && immersed))) {
                                VecDuplicate(user[bi].Csi, &(user[bi].Force_wm)); // added by xyang 1-27-2011
                        }
		
                        //VecDuplicate(user[bi].Csi, &(user[bi].Visc1));
                        //VecDuplicate(user[bi].Csi, &(user[bi].Visc2));
                        //VecDuplicate(user[bi].Csi, &(user[bi].Visc3));

			// end add (xiaolei)



			if(implicit!=4) VecDuplicate(user[bi].Csi, &(user[bi].psuedot));
		//	VecDuplicate(user[bi].Csi, &(user[bi].Area));
		     				     
			VecDuplicate(user[bi].Csi, &(user[bi].Ucont));
			VecDuplicate(user[bi].Csi, &(user[bi].Ucont_o));
			VecDuplicate(user[bi].Csi, &(user[bi].Ucont_rm1));
			//VecDuplicate(user[bi].Csi, &(user[bi].Ucont_o_half));	// seokkoo
			//VecDuplicate(user[bi].Csi, &(user[bi].Ucont_rm2));	// seokkoo

			VecDuplicate(user[bi].Csi, &(user[bi].RHS_o));	// seokkoo
			//VecDuplicate(user[bi].Csi, &(user[bi].RHS_rm1));	// seokkoo
			VecDuplicate(user[bi].Csi, &(user[bi].dP));	// seokkoo
	
			VecDuplicate(user[bi].Csi, &(user[bi].Ucat));
			VecDuplicate(user[bi].Csi, &(user[bi].Ucat_o));
			VecDuplicate(user[bi].Csi, &(user[bi].DUold));
			VecDuplicate(user[bi].Csi, &(user[bi].Bcs.Ubcs));
			VecDuplicate(user[bi].Csi, &(user[bi].GridSpace));
			VecDuplicate(user[bi].Csi, &(user[bi].Itfc));
		//	VecDuplicate(user[bi].Csi, &(user[bi].Normal_I));	// seokkoo
		//	VecDuplicate(user[bi].Csi, &(user[bi].Normal_J));	// seokkoo
		//	VecDuplicate(user[bi].Csi, &(user[bi].Normal_K));	// seokkoo

		//	VecDuplicate(user[bi].Csi, &(user[bi].Rhs));
			if (level < usermg->mglevels-1) {
				VecDuplicate(user[bi].Csi, &(user[bi].Forcing));
				VecDuplicate(user[bi].Csi, &(user[bi].Ucont_MG));
			}

			ierr = DMCreateGlobalVector(user[bi].da, &(user[bi].Aj)); CHKERRQ(ierr);
			VecDuplicate(user[bi].Aj, &(user[bi].P));
			VecDuplicate(user[bi].Aj, &(user[bi].Phi));
			VecDuplicate(user[bi].Aj, &(user[bi].IAj));
			VecDuplicate(user[bi].Aj, &(user[bi].JAj));
			VecDuplicate(user[bi].Aj, &(user[bi].KAj));
			VecDuplicate(user[bi].Aj, &(user[bi].ItfcP));

			VecDuplicate(user[bi].Aj, &(user[bi].Nvert));
			VecDuplicate(user[bi].Aj, &(user[bi].Nvert_o));
			/*
			user[bi].Nvert_tmp = new Vec [NumberOfBodies];
			for(int ibi=0; ibi<NumberOfBodies; ibi++) {
				VecDuplicate(user[bi].Aj, &(user[bi].Nvert_tmp[ibi]));
			}
			*/
			VecDuplicate(user[bi].Aj, &(user[bi].P_o));
			//VecDuplicate(user[bi].Aj, &(user[bi].Volume));
		
			// begin add (xiaolei)
                        if(user->temp) {
                                VecDuplicate(user[bi].Aj, &(user[bi].gTmprt));
                                VecDuplicate(user[bi].Aj, &(user[bi].gTmprt_o));
                                VecDuplicate(user[bi].Aj, &(user[bi].gTmprt_rm1));

                                if (Force_wm && (imin_wmtmprt !=0 || imax_wmtmprt !=0 || jmin_wmtmprt != 0 || jmax_wmtmprt !=0 || (IB_wmtmprt != 0 && immersed))) {
                                        VecDuplicate(user[bi].Aj, &(user[bi].Force_wmtmprt)); // added by xyang 10-22-2012
                                }

	                        //if(les_prt) VecDuplicate(user[bi].Aj, &(user[bi].Pr_t));                     

                        }
                        // end add (xiaolei)

                        if(conv_diff)
                        {
                          VecDuplicate(user[bi].Aj, &(user[bi].Conc));
                          VecDuplicate(user[bi].Aj, &(user[bi].Conc_o));
                        }
	
			DMCreateLocalVector(user[bi].fda, &(user[bi].lCsi));

			VecDuplicate(user[bi].lCsi, &(user[bi].lEta));
			VecDuplicate(user[bi].lCsi, &(user[bi].lZet));
			VecDuplicate(user[bi].lCsi, &(user[bi].lICsi));
			VecDuplicate(user[bi].lCsi, &(user[bi].lIEta));
			VecDuplicate(user[bi].lCsi, &(user[bi].lIZet));
			VecDuplicate(user[bi].lCsi, &(user[bi].lJCsi));
			VecDuplicate(user[bi].lCsi, &(user[bi].lJEta));
			VecDuplicate(user[bi].lCsi, &(user[bi].lJZet));
			VecDuplicate(user[bi].lCsi, &(user[bi].lKCsi));
			VecDuplicate(user[bi].lCsi, &(user[bi].lKEta));
			VecDuplicate(user[bi].lCsi, &(user[bi].lKZet));
			VecDuplicate(user[bi].lCsi, &(user[bi].lGridSpace));
			VecDuplicate(user[bi].lCsi, &(user[bi].lCent));
			
			
			// begin add (xiaolei)
                        // added by xyang 12-7-2010
                        if (rotor_model || IB_delta)
                          VecDuplicate(user[bi].lCsi, &(user[bi].lF_eul));

                        // added by xyang 
                        if (user->temp)
                          VecDuplicate(user[bi].lCsi, &(user[bi].lFTmprt));

                        // added by xyang 
                        if (conv_diff)
                          VecDuplicate(user[bi].lCsi, &(user[bi].lFCurrent));

			
                        if (Force_wm && (imin_wm !=0 || imax_wm !=0 || jmin_wm != 0 || jmax_wm !=0 || (IB_wm && immersed)))
                          VecDuplicate(user[bi].lCsi, &(user[bi].lForce_wm)); // added by xyang 12-7-2010

                        if ((imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || (IB_wm && immersed)) && Shear_wm) {
                                VecDuplicate(user[bi].lCsi, &(user[bi].lVisc1_wm));
                                VecDuplicate(user[bi].lCsi, &(user[bi].lVisc2_wm));
                                VecDuplicate(user[bi].lCsi, &(user[bi].lVisc3_wm));

                        }

                        VecDuplicate(user[bi].lCsi, &(user[bi].lVisc1));
                        VecDuplicate(user[bi].lCsi, &(user[bi].lVisc2));
                        VecDuplicate(user[bi].lCsi, &(user[bi].lVisc3));

			// end add (xiaolei)

				 
			VecDuplicate(user[bi].lCsi, &(user[bi].lUcont));
			VecDuplicate(user[bi].lCsi, &(user[bi].lUcat));
			VecDuplicate(user[bi].lCsi, &(user[bi].lItfc));

			VecDuplicate(user[bi].lCsi, &(user[bi].lUcat_old));	// seokkoo
	
			
			VecDuplicate(user[bi].lCsi, &user[bi].Fp);
			VecDuplicate(user[bi].lCsi, &user[bi].Div1);
			VecDuplicate(user[bi].lCsi, &user[bi].Div2);
			VecDuplicate(user[bi].lCsi, &user[bi].Div3);
			//VecDuplicate(user[bi].lCsi, &user[bi].Visc1);
			//VecDuplicate(user[bi].lCsi, &user[bi].Visc2);
			//VecDuplicate(user[bi].lCsi, &user[bi].Visc3);
			
			//VecDuplicate(user[bi].lCsi, &user[bi].lUcont_c);

			//VecDuplicate(user[bi].lCsi, &(user[bi].Conv_o));	// seokkoo
			//VecDuplicate(user[bi].lCsi, &(user[bi].Visc_o));		// seokkoo
			//VecDuplicate(user[bi].lCsi, &(user[bi].lUcat_i));		// seokkoo
			//VecDuplicate(user[bi].lCsi, &(user[bi].lUcat_j));		// seokkoo
			//VecDuplicate(user[bi].lCsi, &(user[bi].lUcat_k));		// seokkoo
	
	
      
			VecDuplicate(user[bi].lCsi, &(user[bi].lUcont_o));
			VecDuplicate(user[bi].lCsi, &(user[bi].lUcont_rm1));
			//VecDuplicate(user[bi].lCsi, &(user[bi].lArea));

			DMCreateLocalVector(user[bi].da, &(user[bi].lAj));
      
			VecDuplicate(user[bi].lAj, &(user[bi].lIAj));
			VecDuplicate(user[bi].lAj, &(user[bi].lJAj));
			VecDuplicate(user[bi].lAj, &(user[bi].lKAj));
			VecDuplicate(user[bi].lAj, &(user[bi].lP));
			VecDuplicate(user[bi].lAj, &(user[bi].lPhi));
			VecDuplicate(user[bi].lAj, &(user[bi].lItfcP));

			VecDuplicate(user[bi].lAj, &(user[bi].lNvert_o_fixed));
			VecSet(user[bi].lNvert_o_fixed, 0);

			user[bi].lNvert_tmp = new Vec [NumberOfBodies];
                        for(int ibi=0; ibi<NumberOfBodies; ibi++) {
			  VecDuplicate(user[bi].lAj, &(user[bi].lNvert_tmp[ibi]));
                        }

			VecDuplicate(user[bi].lAj, &(user[bi].lNvert));
			VecDuplicate(user[bi].lAj, &(user[bi].lNvert_o));

			//VecDuplicate(user[bi].lAj, &(user[bi].lVolume));

                        // add begin (xiaolei)
                        if (user->temp) {
                                VecDuplicate(user[bi].lAj, &(user[bi].lTmprt));
                                VecDuplicate(user[bi].lAj, &(user[bi].lTmprt_o));
                                VecDuplicate(user[bi].lAj, &(user[bi].lTmprt_rm1));

                                if (Force_wm && (imin_wmtmprt !=0 || imax_wmtmprt !=0 || jmin_wmtmprt != 0 || jmax_wmtmprt !=0 || (IB_wmtmprt != 0 && immersed))) {
                                        VecDuplicate(user[bi].lAj, &(user[bi].lForce_wmtmprt)); // added by xyang 10-22-2012
				}

                                if (Shear_wm && (imin_wmtmprt !=0 || imax_wmtmprt !=0 || jmin_wmtmprt != 0 || jmax_wmtmprt !=0 || (IB_wmtmprt != 0 && immersed))) {
                                        VecDuplicate(user[bi].lAj, &(user[bi].lVisc1_wmtmprt));
                                        VecDuplicate(user[bi].lAj, &(user[bi].lVisc2_wmtmprt));
                                        VecDuplicate(user[bi].lAj, &(user[bi].lVisc3_wmtmprt));
                                }

                                VecDuplicate(user[bi].lAj, &(user[bi].lVisc1_tmprt));
                                VecDuplicate(user[bi].lAj, &(user[bi].lVisc2_tmprt));
                                VecDuplicate(user[bi].lAj, &(user[bi].lVisc3_tmprt));


                        	if (user->temp && les_prt) {
                                	VecDuplicate(user[bi].lAj, &(user[bi].lPr_t));
                        	}

                        }
                        // add end (xiaolei)

                        if (conv_diff)
                        {
                          VecDuplicate(user[bi].lAj, &(user[bi].lConc));
                          VecDuplicate(user[bi].lAj, &(user[bi].lConc_o));
                        }
 
			if(rans) {
				DMCreateLocalVector(user[bi].fda2, &user[bi].lK_Omega);	VecSet(user[bi].lK_Omega, 0);	// seokkoo
				DMCreateLocalVector(user[bi].fda2, &user[bi].lK_Omega_o);	VecSet(user[bi].lK_Omega_o, 0);	// seokkoo
				VecDuplicate(user[bi].P, &user[bi].Distance);
	
				DMCreateGlobalVector(user[bi].fda2, &user[bi].K_Omega);	VecSet(user[bi].K_Omega, 0);	// seokkoo
				VecDuplicate(user[bi].K_Omega, &(user[bi].K_Omega_o));	VecSet(user[bi].K_Omega_o, 0);// seokkoo
				//VecDuplicate(user[bi].K_Omega, &(user[bi].K_Omega_rm1));
				//VecDuplicate(user[bi].lP, &(user[bi].lSrans));		VecSet(user[bi].lSrans, 0);// seokkoo
				VecDuplicate(user[bi].lP, &(user[bi].lNu_t));		VecSet(user[bi].lNu_t, 0);// seokkoo
				
		
				if(rans==3) {
					VecDuplicate(user[bi].lP, &(user[bi].lF1));
					VecSet(user[bi].lF1, 0);
				}
			}
	
			if(les) {
				/*
				VecDuplicate(user->P, &user->Cs);
				VecDuplicate(user->P, &user->Cs_o);*/
				VecDuplicate(user->lP, &user->lCs);
				/*VecDuplicate(user->lP, &user->lCs_o);*/
				/*VecDuplicate(user->lUcont, &user->lCs_IJK);
				VecDuplicate(user->lUcont, &user->lNu_t_IJK);*/
				
				VecDuplicate(user[bi].lP, &(user[bi].lNu_t));		
				VecSet(user[bi].lNu_t, 0);
			}
			
			if(levelset) {
				Initialize_free_surface_location_vector(&user[bi]);
                                //VecDuplicate(user[bi].lP, &user[bi].lLevelset);       VecSet(user->lLevelset, 0);
				VecDuplicate(user[bi].lUcont, &user[bi].lST);       VecSet(user[bi].lST, 0);
                                VecDuplicate(user[bi].P, &user[bi].Levelset); VecSet(user[bi].Levelset, 0);
				VecDuplicate(user[bi].P, &user[bi].Levelset_o);       VecSet(user[bi].Levelset_o, 0);
                                VecDuplicate(user[bi].lP, &user[bi].lDensity);        VecSet(user[bi].lDensity, 0);
                                VecDuplicate(user[bi].lP, &user[bi].lMu);     VecSet(user[bi].lMu, 0);

				VecDuplicate(user[bi].lUcont, &user[bi].lFSponge);       VecSet(user[bi].lFSponge, 0); // xiaolei add for sponge layer at the outlet
				VecDuplicate(user[bi].Ucont, &user[bi].FSponge);       VecSet(user[bi].FSponge, 0); // xiaolei add for sponge layer at the outlet

				VecDuplicate(user[bi].lUcont, &user[bi].Curv);        // xiaolei Store the level set curvature
				VecDuplicate(user[bi].lP, &user[bi].Grad_abs);        // xiaolei Store the level set gradient abs
				VecDuplicate(user[bi].lP, &user[bi].Heaviside);        // xiaolei Store the level set Heaviside
                        }
			if(rans==3 || levelset) {
			  VecDuplicate(user[bi].lP, &user[bi].lLevelset);       VecSet(user->lLevelset, 0);
			}


			// PetscPrintf(PETSC_COMM_WORLD, "test\n");
			FormInitialize(&(user[bi]));

			PetscPrintf(PETSC_COMM_WORLD, "Initialization\n");

			FormMetrics(&(user[bi]));
		}
	}
	return 0;
}


PetscErrorCode MG_Finalize(UserMG *usermg)
{

  MGCtx *mgctx;

  int level, bi;

  UserCtx *user;

  mgctx = usermg->mgctx;
  for (level=usermg->mglevels-1; level>=0; level--) {
    user=mgctx[level].user;
    for (bi=0; bi<block_number; bi++) {
	
	if(level==usermg->mglevels-1 && averaging) {
		VecDestroy(&user[bi].Ucat_square_sum);
		VecDestroy(&user[bi].Ucat_cross_sum);
		VecDestroy(&user[bi].Ucat_sum);
	}
      VecDestroy(&user[bi].Cent);
      VecDestroy(&user[bi].Ucont);
      VecDestroy(&user[bi].Ucont_o);
      VecDestroy(&user[bi].Ucont_rm1);
      //VecDestroy(&user[bi].Ucont_rm2);	// seokkoo
      VecDestroy(&user[bi].Ucat);
      VecDestroy(&user[bi].Ucat_o);
      VecDestroy(&user[bi].DUold);
      VecDestroy(&user[bi].Bcs.Ubcs);
      VecDestroy(&user[bi].GridSpace);
      VecDestroy(&user[bi].Itfc);
      //      VecDestroy(&user[bi].Rhs);
      if(implicit!=4) VecDestroy(&user[bi].psuedot);
      //VecDestroy(&user[bi].Area);
      //VecDestroy(&user[bi].Volume);

	// add begin (xiaolei)
      	if (rotor_model || IB_delta) VecDestroy(&user[bi].F_eul); // xyang 12-7-2010
      	if (Force_wm && (imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || (IB_wm != 0 && immersed))) {
		VecDestroy(&user[bi].Force_wm); // xyang 12-7-2010
      	}

	if (Force_wm && (user->temp && (imin_wmtmprt !=0 || imax_wmtmprt !=0 || jmin_wmtmprt != 0 || jmax_wmtmprt !=0 || (IB_wmtmprt != 0 && immersed)))) {
		VecDestroy(&user[bi].Force_wmtmprt); // added by xyang 10-22-2012
	}


	if (user->temp) {
	      //VecDestroy(&user[bi].Visc1_tmprt);
	      //VecDestroy(&user[bi].Visc2_tmprt);
	      //VecDestroy(&user[bi].Visc3_tmprt);
	}

      	if (user->temp)
          VecDestroy(&user[bi].gFTmprt); // added by xyang 

      	if (conv_diff)
          VecDestroy(&user[bi].FCurrent); // added by xyang 

	// add end (xiaolei)


      if (level < usermg->mglevels-1) {
	VecDestroy(&user[bi].Forcing);
	VecDestroy(&user[bi].Ucont_MG);
      }

      VecDestroy(&user[bi].Nvert);
      VecDestroy(&user[bi].Nvert_o);
     
      VecDestroy(&user[bi].P);
      VecDestroy(&user[bi].Phi);
      VecDestroy(&user[bi].P_o);

      VecDestroy(&user[bi].lCsi);
      VecDestroy(&user[bi].lEta);
      VecDestroy(&user[bi].lZet);
      VecDestroy(&user[bi].lICsi);
      VecDestroy(&user[bi].lIEta);
      VecDestroy(&user[bi].lIZet);
      VecDestroy(&user[bi].lJCsi);
      VecDestroy(&user[bi].lJEta);
      VecDestroy(&user[bi].lJZet);
      VecDestroy(&user[bi].lKCsi);
      VecDestroy(&user[bi].lKEta);
      VecDestroy(&user[bi].lKZet);
      VecDestroy(&user[bi].lGridSpace);
      VecDestroy(&user[bi].lUcont);
      VecDestroy(&user[bi].lUcat);
      VecDestroy(&user[bi].ItfcP);
      VecDestroy(&user[bi].lCent);
	// add begin (xiaolei)
      	if (rotor_model || IB_delta) VecDestroy(&user[bi].lF_eul); // xyang 12-7-2010
      	if (Force_wm && (imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm !=0 || (IB_wm && immersed))) {

		VecDestroy(&user[bi].lForce_wm); // xyang 12-7-2010

	}
        if ((imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || (IB_wm && immersed)) && Shear_wm) {
                VecDestroy(&user[bi].lVisc1_wm);
                VecDestroy(&user[bi].lVisc2_wm);
                VecDestroy(&user[bi].lVisc3_wm);
      	}


	if (Force_wm && (imin_wmtmprt !=0 || imax_wmtmprt !=0 || jmin_wmtmprt != 0 || jmax_wmtmprt !=0 || (IB_wmtmprt != 0 && immersed))) {
		VecDestroy(&user[bi].lForce_wmtmprt); // added by xyang 10-22-2012
	}


	if (Shear_wm && (imin_wmtmprt !=0 || imax_wmtmprt !=0 || jmin_wmtmprt != 0 || jmax_wmtmprt !=0 || (IB_wmtmprt != 0 && immersed))) {
                VecDestroy(&user[bi].lVisc1_wmtmprt);
                VecDestroy(&user[bi].lVisc2_wmtmprt);
                VecDestroy(&user[bi].lVisc3_wmtmprt);
	}



      	//VecDestroy(&user[bi].lVisc1);
      	//VecDestroy(&user[bi].lVisc2);
      	//VecDestroy(&user[bi].lVisc3);

	if (user->temp) {
      		VecDestroy(&user[bi].lVisc1_tmprt);
      		VecDestroy(&user[bi].lVisc2_tmprt);
      		VecDestroy(&user[bi].lVisc3_tmprt);
	}
	// add end (xiaolei)

      VecDestroy(&user[bi].lUcont_o);
      VecDestroy(&user[bi].lUcont_rm1);
//      VecDestroy(&user[bi].lArea);
//      VecDestroy(&user[bi].lVolume);

      VecDestroy(&user[bi].lAj);
      VecDestroy(&user[bi].lIAj);
      VecDestroy(&user[bi].lJAj);
      VecDestroy(&user[bi].lKAj);
      VecDestroy(&user[bi].lP);
      VecDestroy(&user[bi].lPhi);
      VecDestroy(&user[bi].lNvert);
      VecDestroy(&user[bi].lNvert_o);
      VecDestroy(&user[bi].lItfc);
      VecDestroy(&user[bi].lItfcP);

      DMDestroy(&user[bi].da);
/*       DMDestroy(&user[bi].fda); */
    }
    PetscFree(user);
  }
  PetscFree(usermg->mgctx);
  //  PetscFree(usermg);
  return 0;
}
