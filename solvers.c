#include "variables.h"

void Init_LevelSet_Vectors(UserCtx *user);
void Destroy_LevelSet_Vectors(UserCtx *user);
void Distance_Function_IC(UserCtx *user);
void Solve_Distance(UserCtx *user);


extern void Solve_Conv_Diff(UserCtx *user, double & norm);
extern void Conv_Diff_IC(UserCtx *user);
extern void Force_current(UserCtx *user);
extern int conv_diff,density_current;


PetscErrorCode Struc_Solver(UserMG *usermg,IBMNodes *ibm, 
			    FSInfo *fsi, int itr_sc,
			    PetscBool *DoSCLoop)
{
  PetscReal     dS_sc, dS_MIN=1e-5, dSmax;
  UserCtx	*user;
  PetscInt i,bi,ibi, level, Add_dUndt=1,MHV_stuck=0 ;

  level = usermg->mglevels-1;
  user = usermg->mgctx[level].user;

/* ==================================================================================             */
/*     Store old values to determine SC convergence */

  if (movefsi || rotatefsi || MHV || fish || cop) {
    for (ibi=0;ibi<NumberOfBodies;ibi++) {
    for (i=0;i<6;i++){
      fsi[ibi].S_old[i] = fsi[ibi].S_new[i];
      fsi[ibi].S_ang_o[i]=fsi[ibi].S_ang_n[i];      
      if (itr_sc==1) {
	fsi[ibi].dS[i]=0.;
	fsi[ibi].atk=0.3;	
      }
      fsi[ibi].dS_o[i]=fsi[ibi].dS[i];
      fsi[ibi].atk_o=fsi[ibi].atk;
    }
    if (itr_sc==2)
      fsi[ibi].atk_o=0.298;

    fsi[ibi].F_x_old=fsi[ibi].F_x;
    fsi[ibi].F_y_old=fsi[ibi].F_y;
    fsi[ibi].F_z_old=fsi[ibi].F_z;
    
    fsi[ibi].M_x_old=fsi[ibi].M_x;
    fsi[ibi].M_y_old=fsi[ibi].M_y;
    fsi[ibi].M_z_old=fsi[ibi].M_z;
    
    }
  }
  
/* ==================================================================================             */
/*     Calculating Forces! */
  if (MHV) Add_dUndt=0;

  if (immersed) {
    for (bi=0; bi<block_number; bi++) {      
      for (ibi=0;ibi<NumberOfBodies;ibi++) {
      if (!sediment )Calc_forces_SI(&fsi[ibi],&(user[bi]),&ibm[ibi], ti, ibi, bi); /// SEDI   
      }//ibi
/* ==================================================================================             */
/*       Ucat is copied here before it is changed by the flow solver */
/*       it is needed for calculating the forces by CV method */
/*       Ucat_o shouldn't be equal to Ucat  */

      // Copy Ucat_o here!
      if (itr_sc==1) VecCopy(user[bi].Ucat, user[bi].Ucat_o);

      /* Corrector step! start from the same sol  */

      if ((sediment || MHV || movefsi || rotatefsi || fish) && itr_sc>1) {  // SEDI
	PetscPrintf(PETSC_COMM_WORLD, "Corrector Step itr # %d\n", itr_sc);
    
	VecCopy(user[bi].Ucont_o, user[bi].Ucont);
	VecCopy(user[bi].P_o, user[bi].P);

	DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
	DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);

	DMGlobalToLocalBegin(user[bi].da, user[bi].P, INSERT_VALUES, user[bi].lP);
	DMGlobalToLocalEnd(user[bi].da, user[bi].P, INSERT_VALUES, user[bi].lP);

	Contra2Cart(&(user[bi]));
      }
      PetscBarrier(PETSC_NULL);	  
    }
  }

/* ==================================================================================             */
/*     Find The new Position & Move the BODY */
  
  if (movefsi){
    for (level = usermg->mglevels-1; level>=usermg->mglevels-1; level--) {
      user = usermg->mgctx[level].user;
      for (bi=0; bi<block_number; bi++) {
	if (immersed) {
	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  Calc_FSI_pos_SC(&fsi[ibi], &ibm[ibi], 0.5*(user->dt), user->dt, user->ren) ;
	  //Forced_Motion(&fsi[ibi], 0.5,user->dt);	  
	}

	CollisionDetectionOfCylinders(fsi,NumberOfBodies);

	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  Elmt_Move_FSI_TRANS(&fsi[ibi], &ibm[ibi]);
	}
	PetscBarrier(PETSC_NULL);

	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA\n");
	  ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
	  PetscBarrier(PETSC_NULL);	  

	}
	}
      }
    }
  }
  
  if (rotatefsi) { /*already did search at ti=tistart*/
    for (level = usermg->mglevels-1; level>=usermg->mglevels-1; level--) {
      user = usermg->mgctx[level].user;
      for (bi=0; bi<block_number; bi++) {
	if (immersed) {
	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  //Calc_FSI_Ang(&fsi[ibi], &ibm[ibi], 0.5*(user->dt), user->dt,ibi,&user[bi]) ;
	  //Forced_Rotation(fsi, 20*pi0180,2.*pi*0.185*0.97,user->dt);
		fsi[ibi].x_c = x_r;
		fsi[ibi].y_c = y_r;
		fsi[ibi].z_c = z_r;
		if(ibi==0 || ibi<NumberOfRotatingBodies) {
		  Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi);
		}
	  //SwingCylinder(&fsi[ibi], &ibm[ibi]);
		  PetscBarrier(PETSC_NULL);
		  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA solver.c begin\n");
		  ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
		  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA solver.c end\n");
		  PetscBarrier(PETSC_NULL);		
	}
	}
      }
    }
  }

  if (MHV && ti>-1) {
    for (level = usermg->mglevels-1; level>=usermg->mglevels-1; level--) {
      user = usermg->mgctx[level].user;
      PetscReal dir=1.;

      for (bi=0; bi<block_number; bi++) {
	if (immersed) {
	  // for leaflets ibi = 1 & 2
	for (ibi=1;ibi<NumberOfBodies;ibi++) {
	  dir = -1*dir;
/* 	for (itr_dUndt=0; itr_dUndt<10;itr_dUndt++) { */
/* 	  fsi[ibi].S_ang_o[1]=fsi[ibi].S_ang_n[1]; */

/* 	  if (STRONG_COUPLING)  */
/* 	  if (itr_sc==1) { */
/* 	  fsi[ibi].S_ang_n[1]=2.*fsi[ibi].S_ang_r[1]-fsi[ibi].S_ang_rm1[1]; */
/* 	  fsi[ibi].S_ang_n[0]=fsi[ibi].S_ang_r[0]+0.5*(fsi[ibi].S_ang_n[1]+fsi[ibi].S_ang_r[1])*user->dt; */
/* 	  } else { */
	  Calc_FSI_Ang_intg(&fsi[ibi], &ibm[ibi], user->dt,itr_sc,ibi,&user[bi]) ;
/* 	  } */
/* 	  else */
/* 	  Calc_FSI_Ang_staggered(&fsi[ibi], &ibm[ibi], user->dt,ibi,&user[bi]) ; */

	  if ((dir*fsi[ibi].S_ang_n[0])> -max_angle) {
	    fsi[ibi].S_ang_n[0]= -dir*max_angle;
	    fsi[ibi].S_ang_n[1]= 0.;
	    MHV_stuck=1;
	  }
	  if ((dir*fsi[ibi].S_ang_n[0])< 0.0) {
	    fsi[ibi].S_ang_n[0]= dir*0.0;
	    fsi[ibi].S_ang_n[1]= 0.;
	    MHV_stuck=1;
	  }

/* /\* 	  Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi); *\/ */

/* /\* 	  PetscBarrier(PETSC_NULL); *\/ */

/* /\* 	  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA\n"); *\/ */
/* /\* 	  ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi); *\/ */

/* 	  for (i=0; i<ibm[ibi].n_v; i++) { */
/* 	    rx = ibm[ibi].x_bp[i]-fsi[ibi].x_c; */
/* 	    ry = ibm[ibi].y_bp[i]-fsi[ibi].y_c; */
/* 	    rz = ibm[ibi].z_bp[i]-fsi[ibi].z_c; */
/* 	    ibm[ibi].u[i].x =   ry*fsi[ibi].S_ang_n[5]-fsi[ibi].S_ang_n[3]*rz  ; */
/* 	    ibm[ibi].u[i].y =-( rx*fsi[ibi].S_ang_n[5]-fsi[ibi].S_ang_n[1]*rz ); */
/* 	    ibm[ibi].u[i].z =   rx*fsi[ibi].S_ang_n[3]-fsi[ibi].S_ang_n[1]*ry  ; */
/* 	  } */

/* 	  PetscBarrier(PETSC_NULL); */
/* 	  ibm_interpolation_advanced(&user[bi], &ibm[ibi], &fsi[ibi], ibi,0); */
/* 	  PetscBarrier(PETSC_NULL); */
/* 	  Calc_forces_SI(&fsi[ibi],&(user[bi]),&ibm[ibi], ti, ibi, bi); */

/* 	  PetscPrintf(PETSC_COMM_WORLD, "FSI convergence OL %d w_x:%le\n",ibi, fsi[ibi].S_ang_n[1]-fsi[ibi].S_ang_o[1]); */
  
/* 	} */

	}
	//PetscBarrier(PETSC_NULL);
	for (ibi=1;ibi<NumberOfBodies;ibi++) {
	  Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi);
	}
	PetscBarrier(PETSC_NULL);
	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA\n");
	  ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
	  PetscBarrier(PETSC_NULL);	  
	}
	}
      }
    }
  }

  
 if (sediment && immersed) {
	PetscReal ts, te, cputime;
    	level = usermg->mglevels-1;
    	user = usermg->mgctx[level].user;
    	for (bi=0; bi<block_number; bi++) {
		for (ibi=0;ibi<1;ibi++) {
		        PetscGetTime(&ts);
		        Scour(&user[bi],&ibm[ibi],tistart,ti,itr_sc);
		        PetscGetTime(&te);
		        cputime=te-ts;
		        PetscPrintf(PETSC_COMM_WORLD, "Total Sediment-Transport cputime %d %le\n",ti,cputime);

			calc_ibm_volumeFlux(&ibm[ibi], user[bi].dt, &(user[bi].FluxIntpSum));
		}

		if (ti == (ti/bed_avg)*bed_avg) {
			VecSet(user[bi].Nvert,0.);
			VecSet(user[bi].lNvert,0.);
			for (ibi=0;ibi<1;ibi++) {
	  			PetscPrintf(PETSC_COMM_WORLD, "IBM_Search\n");
		   	     	PetscGetTime(&ts);
			  	ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
	
		  		PetscBarrier(PETSC_NULL);	  
        			PetscGetTime(&te);
        			cputime=te-ts;
	        		PetscPrintf(PETSC_COMM_WORLD, "IBMSEARCH cputime %d %le\n",ti,cputime);
        			//ibm_interpolation_advanced(&(user[bi]),&ibm[ibi],ibi,1);
      
        			ibm_interpolation_advanced(&(user[bi]),1);
       				PetscPrintf(PETSC_COMM_WORLD, "IBM_Interpolation done after bed changed/\n");
	          	}
		}

    	}// bi
  }


/* ==================================================================================             */
/*   Convergence of the SC Loop */
/* ==================================================================================             */

  *DoSCLoop = PETSC_FALSE;
  dSmax=1e-10;
  for (ibi=0;ibi<NumberOfBodies;ibi++) {
  
  if ((movefsi || fish) && STRONG_COUPLING) {
    for (i=0;i<6;i++) {
    dS_sc = fabs(fsi[ibi].S_new[i]-fsi[ibi].S_old[i]);
    if (dS_sc > dS_MIN) *DoSCLoop = PETSC_TRUE;
    if (dS_sc > dSmax) dSmax=dS_sc;
    }
/*     dS_sc = fabs(fsi[ibi].S_new[2]-fsi[ibi].S_old[2]); */
/*     if (dS_sc > dS_MIN) *DoSCLoop = PETSC_TRUE; */
  }

  if ((rotatefsi||MHV) && STRONG_COUPLING) {
    dS_sc = fabs(fsi[ibi].S_ang_n[0]-fsi[ibi].S_ang_o[0]);
    if (dS_sc > dS_MIN) *DoSCLoop = PETSC_TRUE;
    dS_sc = fabs(fsi[ibi].S_ang_n[1]-fsi[ibi].S_ang_o[1]);
    if(fabs(fsi[ibi].S_ang_n[1]+fsi[ibi].S_ang_o[1])>2.) 
      dS_sc /= 0.5*fabs(fsi[ibi].S_ang_n[1]+fsi[ibi].S_ang_o[1]);

    if (dS_sc > dS_MIN) *DoSCLoop = PETSC_TRUE;
    if (dS_sc > dSmax) dSmax=dS_sc;

/*     *DoSCLoop = PETSC_TRUE; */
  }


  //*******************************************
 if (sediment && STRONG_COUPLING)
    {
       double DZ= 0.0;
       int iter=0.0;
     for (i=0;i<ibm[ibi].n_v;i++)
         {
          if (ibm->nf_z[i]<1.e-7 || ibm->elmt_depth[i]>0.2)
             {
             } 
          else
             {
             // DZ = PetscMax(DZ, fabs(ibm[ibi].z_bp_l[i]-ibm[ibi].z_bp[i]));
              DZ += (ibm[ibi].z_bp_l[i]-ibm[ibi].z_bp[i])*(ibm[ibi].z_bp_l[i]-ibm[ibi].z_bp[i]);
              iter++;
             }
        }
      DZ = sqrt(DZ/iter);
      if (DZ > dS_MIN) *DoSCLoop = PETSC_TRUE;
     
      PetscPrintf(PETSC_COMM_WORLD, "bed_change SC Convergence ti & itr_sc & residual %d %d %le \n", ti, itr_sc, DZ);
    }



  
  if ((movefsi || rotatefsi || MHV || fish || sediment) && STRONG_COUPLING && itr_sc<2) *DoSCLoop = PETSC_TRUE;  

  } // ibi  

  if (itr_sc>10) *DoSCLoop = PETSC_FALSE;
  if (MHV_stuck && itr_sc==1) *DoSCLoop = PETSC_FALSE;

  
  PetscPrintf(PETSC_COMM_WORLD, "S-C Convergence %d %le %le %le\n", itr_sc, dSmax,fsi[1].S_ang_n[1],fsi[1].S_ang_o[1]);
  
  for (ibi=0;ibi<NumberOfBodies;ibi++) {

    if (ti == (ti/tiout) * tiout && (movefsi || rotatefsi || cop || MHV) && !(*DoSCLoop)) FSI_DATA_Output(&fsi[ibi], ibi);
  
  }
/* ==================================================================================             */
    
  return(0);
}
/* ==================================================================================             */

PetscErrorCode Struc_predictor(UserMG *usermg,IBMNodes *ibm, 
			       FSInfo *fsi, int itr_sc,
			       int tistart, 
			       PetscBool *DoSCLoop)
{
  UserCtx	*user;
  int	bi,ibi, level, MHV_stuck=0 ;

  level = usermg->mglevels-1;
  user = usermg->mgctx[level].user;

  if (MHV && ti>-1) {
    for (level = usermg->mglevels-1; level>=usermg->mglevels-1; level--) {
      user = usermg->mgctx[level].user;
      PetscReal dir=1.;

      for (bi=0; bi<block_number; bi++) {
	if (immersed) {
	  // for leaflets ibi = 1 & 2
	for (ibi=1;ibi<NumberOfBodies;ibi++) {
	  dir = -1*dir;
	  fsi[ibi].S_ang_n[1]=2.*fsi[ibi].S_ang_r[1]-fsi[ibi].S_ang_rm1[1];
	  fsi[ibi].S_ang_n[0]=fsi[ibi].S_ang_r[0]+0.5*(fsi[ibi].S_ang_n[1]+fsi[ibi].S_ang_r[1])*user->dt;

	  if ((dir*fsi[ibi].S_ang_n[0])> -max_angle) {
	    fsi[ibi].S_ang_n[0]= -dir*max_angle;
	    fsi[ibi].S_ang_n[1]= 0.;
	    MHV_stuck=1;
	  }
	  if ((dir*fsi[ibi].S_ang_n[0])< 0.0) {
	    fsi[ibi].S_ang_n[0]= dir*0.0;
	    fsi[ibi].S_ang_n[1]= 0.;
	    MHV_stuck=1;
	  }


	}
	//PetscBarrier(PETSC_NULL);
	for (ibi=1;ibi<NumberOfBodies;ibi++) {
	  Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi);
	}
	PetscBarrier(PETSC_NULL);
	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA\n");
	  ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
	  PetscBarrier(PETSC_NULL);	  
	}
	}
      }
    }
  }    
  return(0);
}
/* ==================================================================================             */


/* ==================================================================================             */
/*     Flow Solver! */
/* ==================================================================================             */
PetscErrorCode Flow_Solver(UserMG *usermg,IBMNodes *ibm, FSInfo *fsi, 
IBMNodes *wtm, ACL *acl, FSInfo *fsi_wt, IBMNodes *ibm_ACD, FSInfo *fsi_IBDelta,IBMNodes *ibm_IBDelta)
{
  UserCtx	*user;
  int	bi, level;

	
/* ==================================================================================             */

  if (immersed) {
    for (level=usermg->mglevels-1; level>0; level--) {
      for (bi=0; bi<block_number; bi++) {
	MyNvertRestriction(&(usermg->mgctx[level].user[bi]), &(usermg->mgctx[level-1].user[bi]));
      }
    }
  }

/* ==================================================================================             */

  level = usermg->mglevels-1;
  user = usermg->mgctx[level].user;


	
	Calc_Minimum_dt(user);	// momentum.c
  
	#ifdef DIRICHLET
	if(freesurface && ti!=tistart) {
		void update_free_surface_position(UserCtx *user);
		update_free_surface_position(&user[0]);
	}
	#endif

/* ==================================================================================             */
/*   Momentum Solver! */
/* ==================================================================================             */

	if(ti==tistart) {	// seokkoo
		for (bi=0; bi<block_number; bi++) {
			if (immersed && !movefsi && !rotatefsi && ibm_search) {
			  //for (int ibi=0;ibi<NumberOfBodies;ibi++) ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
			}
			//DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
			//DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
			IB_BC(&user[bi]);
			DMLocalToGlobalBegin(user[bi].fda, user[bi].lUcont, INSERT_VALUES, user[bi].Ucont);
			DMLocalToGlobalEnd(user[bi].fda, user[bi].lUcont, INSERT_VALUES, user[bi].Ucont);
			
			//DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat_old);
			//DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat_old);
		}
	} 
	
	#ifdef PRIMITIVE_PRESSURE
	//extern int momentum_option;
	//momentum_option=-1;
	//momentum_option=1;
	#endif 
	
	
		
	if(save_inflow/*pseudo_periodic || k_periodic || kk_periodic*/) {
		/*pseudo_periodic_BC(user);
		if(save_inflow) */
			save_inflow_section(user);
	}
  
	if (inletprofile==100) {	// read inflow data
		read_inflow_section(user);
	}

	/*
	Convection_seokkoo(&user[0], user[0].lUcont, user[0].lUcat, user[0].Conv_o);
	Viscous(&user[0], user[0].lUcont, user[0].lUcat, user[0].Visc_o);
	*/
	
	Calc_k_Flux(&user[0]);
	
	if(les){
		DMGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
		DMGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
		Contra2Cart(user);
		if(ti%dynamic_freq==0 || ti==tistart) Compute_Smagorinsky_Constant_1(user, user->lUcont, user->lUcat);
		Compute_eddy_viscosity_LES(user);
	}
	
	
	if (levelset) {
		Compute_Water_Volume(&user[0], &water_vol_o);
		PetscPrintf(PETSC_COMM_WORLD, "Volume-Prev: %e\n", water_vol_o); 
		/*
		if(ti==tistart && ti==0) {
			Levelset_Function_IC(&user[0]);
		}
		DMGlobalToLocalBegin(user[0].da, user[0].Levelset, INSERT_VALUES, user[0].lLevelset);
		DMGlobalToLocalEnd(user[0].da, user[0].Levelset, INSERT_VALUES, user[0].lLevelset);
		*/
	}

	if(ti==tistart) {
		//PetscPrintf(PETSC_COMM_WORLD, "Start allocating memory for Fp DIV1.. Visc1...\n"); 
		/*
 		VecDuplicate(user[0].lUcont, &user[0].Fp);
		VecDuplicate(user[0].lUcont, &user[0].Div1);
		VecDuplicate(user[0].lUcont, &user[0].Div2);
		VecDuplicate(user[0].lUcont, &user[0].Div3);
		VecDuplicate(user[0].lUcont, &user[0].Visc1);
		VecDuplicate(user[0].lUcont, &user[0].Visc2);
		VecDuplicate(user[0].lUcont, &user[0].Visc3);
		*/
		//PetscPrintf(PETSC_COMM_WORLD, "Finish allocate memory for Fp DIV1.. Visc1...\n"); 
	}
	

	if(implicit==4) {
		/*
		if(levelset && ti==tistart) {
			Compute_Density(&user[0]);
			if(surface_tension) Compute_Surface_Tension(&user[0]);
		}*/
		PetscPrintf(PETSC_COMM_WORLD, "Compute RHS_o\n"); 
		VecSet (user[0].RHS_o, 0.);
		Formfunction_2 (&user[0], user[0].RHS_o, 1.0);
		PetscPrintf(PETSC_COMM_WORLD, "Finish RHS_o\n"); 
        }


	if(levelset) {
		if(!fix_level && ti!=0) {
			//PetscPrintf(PETSC_COMM_WORLD, "Comute LevelSet\n"); 
			double dt = user[0].dt / (double)subdt_levelset;
			for(int t=0; t<subdt_levelset; t++) {
			  //Levelset_smooth(&user[0]);
			  Levelset_BC(&user[0]);
			  Compute_Surface_Curv(&user[0]);
			  //PetscPrintf(PETSC_COMM_WORLD, "Levelset_BC-1 Done.\n");
			  Advect_Levelset(&user[0],dt);
			  //PetscPrintf(PETSC_COMM_WORLD, "Advect_Levelset Done.\n");
			}
			
			  if(smoothlevel) Levelset_smooth(&user[0]); /// just for starting the flow 
			Levelset_BC(&user[0]);
			//PetscPrintf(PETSC_COMM_WORLD, "Levelset_BC Done.\n");
			Reinit_Levelset(&user[0]);
			//PetscPrintf(PETSC_COMM_WORLD, "Reinit_Levelset Done.\n");
			Levelset_BC(&user[0]);
			//Calc_free_surface_location(&user[0]);

			//PetscPrintf(PETSC_COMM_WORLD, "Finish LevelSet\n"); 

			// xiaolei add	
			/*
			char fname[80];
			sprintf(fname,"LEVEL");
			TECIOOut_rhs_da(user, user->Levelset, fname);
			*/

		}
		//PetscPrintf(PETSC_COMM_WORLD, "Compute Density and surface tension\n"); 
		Compute_Surface_Curv(&user[0]); // xiaolei deactivate
		Compute_Density(&user[0]);  // xiaolei deactivate
		if(surface_tension) Compute_Surface_Tension(&user[0]);
		//PetscPrintf(PETSC_COMM_WORLD, "Fnishe Density and surface tension\n"); 
		Compute_Water_Volume(&user[0], &water_vol);
		PetscPrintf(PETSC_COMM_WORLD, "Volume: %e\n", water_vol);
		//Correct_Volume_Flux_Levelset(&user[0]);
		
		//PetscPrintf(PETSC_COMM_WORLD, "Compute Sponge force\n"); 
		if (!fix_level && SpongeLayer) Compute_Force_SpongeLayer(&user[0]);
		
	}
	
	if(rans) {
	  //extern char path[256];
	  char filen[256];
	  PetscViewer     viewer;
	  if( ti==tistart && rans==3 ) {
	    bi=0;
	    sprintf(filen, "%s/Distance_%1d.dat", path, user->_this);
	    if(!ti || !file_exist(filen)) {
	      Compute_Distance_Function(user);
	      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
	      VecView(user->Distance, viewer);
	      PetscViewerDestroy(&viewer);
	    }
	  }

	 
		if(ti==0) {
		  /*
			if(rans==3) {
			  sprintf(filen, "%s/Distance_%1d.dat", path, user->_this);
			  Compute_Distance_Function(user);
			  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
			  VecView(user->Distance, viewer);
			  PetscViewerDestroy(&viewer);
			}
		  */
			PetscPrintf(PETSC_COMM_WORLD, "\nInitializing K-omega ... \n\n");
			K_Omega_IC(user);
			VecSet(user->lNu_t, user->ren);
			
			bi=0;
			VecCopy(user[bi].K_Omega, user[bi].K_Omega_o);
		
			DMGlobalToLocalBegin(user[bi].fda2, user[bi].K_Omega, INSERT_VALUES, user[bi].lK_Omega);
			DMGlobalToLocalEnd(user[bi].fda2, user[bi].K_Omega, INSERT_VALUES, user[bi].lK_Omega);
			DMGlobalToLocalBegin(user[bi].fda2, user[bi].K_Omega_o, INSERT_VALUES, user[bi].lK_Omega_o);
			DMGlobalToLocalEnd(user[bi].fda2, user[bi].K_Omega_o, INSERT_VALUES, user[bi].lK_Omega_o);
		}
		else {
		  bi=0;
		  if(ti==tistart) {
		    VecCopy(user[bi].K_Omega, user[bi].K_Omega_o);
		    DMGlobalToLocalBegin(user[bi].fda2, user[bi].K_Omega, INSERT_VALUES, user[bi].lK_Omega);
		    DMGlobalToLocalEnd(user[bi].fda2, user[bi].K_Omega, INSERT_VALUES, user[bi].lK_Omega);
		    DMGlobalToLocalBegin(user[bi].fda2, user[bi].K_Omega_o, INSERT_VALUES, user[bi].lK_Omega_o);
		    DMGlobalToLocalEnd(user[bi].fda2, user[bi].K_Omega_o, INSERT_VALUES, user[bi].lK_Omega_o);
		  }
		  K_Omega_Set_Constant(user);
		}
		
	}

        if (conv_diff)
        {
	  extern char path[256];
	  char filen[256];
	  PetscViewer     viewer;

          if(ti==0)
          {
            PetscPrintf(PETSC_COMM_WORLD, "\n Initializing Concentration calculations...... \n\n");
            Conv_Diff_IC(user);
                        	
            bi=0;
            VecCopy(user[bi].Conc, user[bi].Conc_o);
		
            DMGlobalToLocalBegin(user[bi].da, user[bi].Conc, INSERT_VALUES, user[bi].lConc);
            DMGlobalToLocalEnd(user[bi].da, user[bi].Conc, INSERT_VALUES, user[bi].lConc);
            DMGlobalToLocalBegin(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
            DMGlobalToLocalEnd(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
          
            if (density_current || user->ctemp)
            {
              VecSet(user[bi].lFCurrent,0);
              VecSet(user[bi].FCurrent,0);
            }

          }
          else
          {
            bi=0;
            if (ti==tistart)
            {
              VecCopy(user[bi].Conc, user[bi].Conc_o);
              DMGlobalToLocalBegin(user[bi].da, user[bi].Conc, INSERT_VALUES, user[bi].lConc);
              DMGlobalToLocalEnd(user[bi].da, user[bi].Conc, INSERT_VALUES, user[bi].lConc);
              DMGlobalToLocalBegin(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
              DMGlobalToLocalEnd(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
            }
          }
	}
	
	
	/*
	VecDuplicate(user[0].lUcont, &user[0].Fp);
	VecDuplicate(user[0].lUcont, &user[0].Div1);
	VecDuplicate(user[0].lUcont, &user[0].Div2);
	VecDuplicate(user[0].lUcont, &user[0].Div3);
	VecDuplicate(user[0].lUcont, &user[0].Visc1);
	VecDuplicate(user[0].lUcont, &user[0].Visc2);
	VecDuplicate(user[0].lUcont, &user[0].Visc3);
	*/

	Pressure_Gradient(&user[0], user[0].dP);

	if (immersed) ibm_interpolation_advanced(&user[0], 1);  // xiaolei deactivate // xiaolei change 0-->1

	// begin add (xiaolei)

	/*
	if (immersed && MoveCylinderTest && MoveFrame) {

		int ibi,i;
		for (ibi=0;ibi<NumberOfBodies;ibi++) { 
  			for (i=0; i<ibm[ibi].n_v; i++) {
    				ibm[ibi].u[i].x = -u_frame;
		    		ibm[ibi].u[i].y = -v_frame;
	    			ibm[ibi].u[i].z = -w_frame;
  			}
		}

	}
	*/

	/*
	char fname[80];
	sprintf(fname,"Nvert");
	TECIOOut_rhs_da(user, user->Nvert, fname);
	*/


  	if (IB_delta || rotor_model) VecSet(user[0].lF_eul,0.0);
  	if (IB_delta || rotor_model) VecSet(user[0].F_eul,0.0);


        if (Force_wm && ( imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || (IB_wm && immersed)))
          VecSet( user[0].lForce_wm,0.0);

        //clock_t start, end;
        //double elapsed;

        //start = clock();

//      ======================================================================================================================
//      rotor model or diffused IB
	PetscReal ts, te, cputime;     // xiaolei
        PetscGetTime(&ts);  // xiaolei

        if (rotor_model) {

		int ibi;
		if (rotor_model==3 && rotatewt) for(ibi=0;ibi<NumberOfTurbines;ibi++) rotor_Rot(&fsi_wt[ibi], &wtm[ibi], user->dt, ibi);
		if (rotor_model==2 && rotatewt) for(ibi=0;ibi<NumberOfTurbines;ibi++) rotor_Rot(&fsi_wt[ibi], &wtm[ibi], user->dt, ibi);

          	if (MoveFrame) {
			UpdateXYZ_MoveFrame (user, wtm, fsi_wt, NumberOfTurbines);
			Pre_process(user, wtm, NumberOfTurbines);
			if (rotor_model == 3) {
                		UpdateXYZ_MoveFrame (user, ibm_ACD, fsi_wt, NumberOfTurbines);
				if (rotor_model==3) Trans1DUp_XYZ(ibm_ACD, NumberOfTurbines);
                		Pre_process(user, ibm_ACD, NumberOfTurbines);
			}
	  	}

		PetscReal ts, te, cputime;     // xiaolei
	        PetscGetTime(&ts);  // xiaolei

		if ((rotor_model==2 || rotor_model==3) && rotatewt) {
			Pre_process(user, wtm, NumberOfTurbines);
		}
 		PetscGetTime(&te);  // xiaolei

	      	PetscPrintf(PETSC_COMM_WORLD, "Time for preprocess of rotor model  %le\n", te-ts);



          	if (rotor_model == 1 ) Calc_F_lagr(user, wtm, fsi_wt, NumberOfTurbines);
	  	if (rotor_model == 3 || rotor_model == 2) Calc_F_lagr_ACL(user, wtm, fsi_wt, acl, NumberOfTurbines);

 	  	if ((rotor_model == 3 || rotor_model == 2) && FixTipSpeedRatio ) Uref_ACL(user, wtm, ibm_ACD, fsi_wt, NumberOfTurbines);

          	for (bi=0; bi<block_number; bi++) {
                	if (rotor_model == 1 ) Calc_forces_rotor(user, wtm, fsi_wt, bi); // xyang 12-08-2010
	               	if (rotor_model == 3 || rotor_model == 2) Calc_forces_ACL(user, wtm, fsi_wt, bi); // xyang 12-08-2010
          	}

          	Calc_F_eul(user, wtm, fsi_wt, NumberOfTurbines);

                PetscGetTime(&te);  // xiaolei

                PetscPrintf(PETSC_COMM_WORLD, "Time for rotor model  %le\n", te-ts);

        }


        if (IB_delta) {
		int ibi;

		if ((rotor_model == 2 || rotor_model==3) && FixTipSpeedRatio ) {
			int ipt;
			int NumLoc=NumberOfIBDelta/NumIBPerLoc;
			if (NumberOfTurbines!=NumLoc) {
      				PetscPrintf(PETSC_COMM_WORLD, "The No. IB_delta locations is inconsistent with Turbine number  \n");
				exit(1);
			} 
			for (ibi=0;ibi<NumLoc;ibi++) 
			for (ipt=0;ipt<NumIBPerLoc;ipt++) { 
				int iname=ibi*NumIBPerLoc+ipt; 
				ibm_IBDelta[iname].U_ref=wtm[ibi].U_ref; 
			}

		}

          	//for(ibi=0;ibi<NumberOfIBDelta;ibi++) PetscPrintf(PETSC_COMM_WORLD, "IBDelta %d U_ref %le\n", ibi, ibm_IBDelta[ibi].U_ref);

 	        Calc_F_lagr_noslip(user, ibm_IBDelta, fsi_IBDelta, NumberOfIBDelta);

                Calc_F_eul(user, ibm_IBDelta, fsi_IBDelta, NumberOfIBDelta);
        }


        //end = clock();
        //elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;


//      ===========================================================================================================
//      wall model

		PetscPrintf(PETSC_COMM_WORLD, "Wall model \n"); 
	if (Force_wm && (imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0))
          Calc_F_wm(user); //0521
	
	if (Force_wm && IB_wm && immersed)
          Calc_F_wmIB(&user[0],ibm); //0521   

	if (Shear_wm && (imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || (IB_wm != 0 && immersed)))
          Visc_wm(user);
	
		PetscPrintf(PETSC_COMM_WORLD, "Finish Wall model \n"); 
	if (user->temp && Shear_wm) {
        	//PetscPrintf(PETSC_COMM_WORLD, "start temperature wall model calculation \n");
		if (imin_wmtmprt != 0 || imax_wmtmprt != 0 || jmin_wmtmprt != 0 || jmax_wmtmprt != 0 || (IB_wmtmprt != 0 && immersed)) Visc_wm_tmprt(user,ibm);
	}

//	================================================================================================================
//	Temperature (scalar equation) 

	if (user->temp)  {
        	PetscPrintf(PETSC_COMM_WORLD, "start temperature calculation \n");
		double start = clock();

		if (les_prt) Compute_Prt(user);

                double tnorm;
                Solve_Tmprt(user, tnorm);

                Force_Tmprt(user);

                for (bi=0; bi<block_number; bi++) {

                        DMGlobalToLocalBegin(user[bi].da, user[bi].gTmprt, INSERT_VALUES, user[bi].lTmprt);
                        DMGlobalToLocalEnd(user[bi].da, user[bi].gTmprt, INSERT_VALUES, user[bi].lTmprt);

                        DMGlobalToLocalBegin(user[bi].fda, user[bi].gFTmprt, INSERT_VALUES, user[bi].lFTmprt);
                        DMGlobalToLocalEnd(user[bi].fda, user[bi].gFTmprt, INSERT_VALUES, user[bi].lFTmprt);
                }

		double end = clock();
	        double elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
	      	PetscPrintf(PETSC_COMM_WORLD, "Time for Temperature %le\n\n", elapsed);
  		int rank, flg=0;
  		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	      	if (!rank) {
			FILE *f;
    			char filen[80];
    			sprintf(filen, "%s/Converge_dU",path);
    			f = fopen(filen, "a");
    			PetscFPrintf(PETSC_COMM_WORLD, f, "%d Temperature %.2e(s) %.5e\n", ti, elapsed, tnorm);
    			fclose(f);
  		}


	}

	// end add (xiaolei)


	if(inletprofile==20){}
	//else if (implicit==1) ImplicitMomentumSolver(user, ibm, fsi);
	//else if (implicit==2) ImplicitMomentumSolver1(user, ibm, fsi);
	//else if (implicit==3) ImpRK(user, ibm, fsi);
	else if (implicit==4) {
	  
        	PetscPrintf(PETSC_COMM_WORLD, "solve momentum equation \n");
	  //  VecSet (user[0].RHS_o, 0.);
	  //	  Formfunction_2 (&user[0], user[0].RHS_o, 1.0);
	 
		
		Implicit_MatrixFree(user, ibm, fsi);  // xiaolei deactivate
	}
  	else {
		COEF_TIME_ACCURACY=1.0;
		RungeKutta(user, ibm, fsi);
	}

	/*
	VecDestroy(&user[0].Fp);
	VecDestroy(&user[0].Div1);
	VecDestroy(&user[0].Div2);
	VecDestroy(&user[0].Div3);
	VecDestroy(&user[0].Visc1);
	VecDestroy(&user[0].Visc2);
	VecDestroy(&user[0].Visc3);
*/
	if(levelset) Correct_Volume_Flux_Levelset(&user[0]);

	
	//if(levelset) Update_Velocity_by_Gravity(&user[0]);
	
	//if (immersed) ibm_interpolation_advanced(&user[0]);
       
	
	
/* ==================================================================================             */
/*    Poisson Solver! */
/* ==================================================================================             */

	PetscBarrier(PETSC_NULL);
    
	
	for(bi=0; bi<block_number; bi++)
        {
          if (inletprofile == 20)
          { }
          /*
          else if (poisson == -1)
            PoissonSolver_MG_original(usermg, ibm, user[bi].ibm_intp);
	  else if (poisson == 0)
            PoissonSolver_MG(usermg, ibm, user[bi].ibm_intp);
          */
          else /*if(poisson==1) */
            PoissonSolver_Hypre(&user[bi], ibm, user[bi].ibm_intp);  // xiaolei deactivate
	}
	
	
	
/* ==================================================================================             */
/*    Velocity Correction! */
/* ==================================================================================             */

	
	if(inletprofile!=20)
          for (bi=0; bi<block_number; bi++)
          {
            UpdatePressure(&user[bi]);  // xiaolei deactivate
            Projection(&(user[bi])); // xiaolei deactivate
            DMGlobalToLocalBegin(user[bi].da, user[bi].P, INSERT_VALUES, user[bi].lP);
            DMGlobalToLocalEnd(user[bi].da, user[bi].P, INSERT_VALUES, user[bi].lP);
          }
	


	//if(levelset) Advect_Levelset(&user[0]);
  
    	#ifdef DIRICHLET
	//free_surafe_BC(&user[0]);
	#endif
  

/* ==================================================================================             */
/*   BC!!    */
/* ==================================================================================             */


  
  if (block_number>1) {
    Block_Interface_U(user);
  }

      //    PetscPrintf(PETSC_COMM_WORLD, "Proj\n");
  for (bi=0; bi<block_number; bi++) {
    if (immersed) {
      /*for (ibi=0;ibi<NumberOfBodies;ibi++) */{
	ibm_interpolation_advanced(&user[bi],1);  // xiaolei deactivate // xiaolei change 1-->0
      }
    }
    

/* ==================================================================================             
   Checking Convergence!
 ==================================================================================             */
    
	bi = 0;	//seokkoo
	Divergence(&(user[bi]));
	
	
	for (bi=0; bi<block_number; bi++) {
		IB_BC(&user[bi]);
		DMLocalToGlobalBegin(user[bi].fda, user[bi].lUcont, INSERT_VALUES, user[bi].Ucont);
		DMLocalToGlobalEnd(user[bi].fda, user[bi].lUcont, INSERT_VALUES, user[bi].Ucont);
		Contra2Cart(&(user[bi]));
	} // 101227

	bi=0;
	Calc_ShearStress(&user[0]);
    
	write_data(&user[0]);

	if (/*(ti == (ti/((int)dt_bed))*((int)dt_bed) || ti==tistart) &&*/ sediment && immersed) write_IBdata(&user[0], ibm);
/* ==================================================================================             */

/*     if (ti == (ti/tiout)*tiout) */
/*       TecOut(&user); */


    //    PetscPrintf(PETSC_COMM_WORLD, "ibm intp\n");
      // Stop out put temporary -lg65

/* ==================================================================================             */
/*     OUTPUT Values! */
/* ==================================================================================             */

	
	if(averaging) {	// seokkoo
		Do_averaging(&user[0]);
	}

	
	KE_Output(user);


	// xiaolei
	if (user->temp) {	
	        extern PetscErrorCode TE_Output(UserCtx *user);
        	TE_Output(user);
	}


	
	int i;
	for (i = 0; i < (rand() % 3000); i++) (rand() % 3000);	//seokkoo

	if(rans) {
		K_Omega_Set_Constant(user);
		Solve_K_Omega(user);
		
//		VecCopy(user[bi].K_Omega_o, user[bi].K_Omega_rm1);
		VecCopy(user[bi].K_Omega, user[bi].K_Omega_o);
		
		DMGlobalToLocalBegin(user[bi].fda2, user[bi].K_Omega, INSERT_VALUES, user[bi].lK_Omega);
		DMGlobalToLocalEnd(user[bi].fda2, user[bi].K_Omega, INSERT_VALUES, user[bi].lK_Omega);
		DMGlobalToLocalBegin(user[bi].fda2, user[bi].K_Omega_o, INSERT_VALUES, user[bi].lK_Omega_o);
		DMGlobalToLocalEnd(user[bi].fda2, user[bi].K_Omega_o, INSERT_VALUES, user[bi].lK_Omega_o);

	}
	
	PetscReal tss,tee,cputimee;	
	if (conv_diff)
        {
          PetscGetTime(&tss);

          double norm = 0.0;
          Solve_Conv_Diff(user, norm);
          VecCopy(user[bi].Conc, user[bi].Conc_o);
		
          DMGlobalToLocalBegin(user[bi].da, user[bi].Conc, INSERT_VALUES, user[bi].lConc);
          DMGlobalToLocalEnd(user[bi].da, user[bi].Conc, INSERT_VALUES, user[bi].lConc);
          DMGlobalToLocalBegin(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
          DMGlobalToLocalEnd(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
            
          PetscGetTime(&tee);
          cputimee = tee-tss;
          int rank;
          MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
          if (!rank)
          {
            FILE *f;
            char filen [80];
            sprintf(filen,"%s/Converge_dU", path);
            f = fopen (filen, "a");
            PetscFPrintf(PETSC_COMM_WORLD, f, "%d(Concentration eqn.) %.2e(s)\n", ti, cputimee);
            PetscFPrintf(PETSC_COMM_WORLD, f, "%d Conc Norm %e\n", ti, norm);
            fclose(f);
          }
	}
        if(density_current || user->ctemp)
          Force_Current(user);
				
				
	
	if(immersed) write_torque_ibm();
	if (ti == (ti/tiout) * tiout || ti==tistart) if(immersed) write_shear_stress_ibm();
	bi=0;
	if (ti == (ti/tiout) * tiout) {
		Ucont_P_Binary_Output(&(user[bi]));  
	}
	else if (tiout_ufield>0 && ti == (ti/tiout_ufield) * tiout_ufield && ti<=tiend_ufield) Ucat_Binary_Output(&(user[bi]));
	
	if(phase_averaging && ti && (ti+ti_lastsave)%phase_averaging==0) Phase_Averaging_Output(&(user[bi]));
  }

/* ==================================================================================             */
/*     End of Flow Solver! */
/* ==================================================================================             */

  return(0);

}
