
PetscErrorCode FormFunction_SNES(SNES snes, Vec Ucont, Vec Rhs, void *ptr) 
{
	UserCtx *user = (UserCtx*)ptr;
	VecCopy(Ucont, user->Ucont);
	
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;
	PetscInt	lxs, lxe, lys, lye, lzs, lze;
	PetscInt	i, j, k;
	
	Cmpnts ***ucont;
	PetscReal ***nvert;
	
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;
	
/*
	DAVecGetArray(user->fda, Ucont, &ucont_1);
        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {


                if (i==3 && j==1 &&  k==3) printf("**** the ucont %le  \n", ucont_1[k][j][i].z );
	}

	DAVecRestoreArray(user->fda, Ucont, &ucont_1);
*/


	DAVecGetArray(user->fda, user->Ucont, &ucont);
	DAVecGetArray(user->da, user->lNvert, &nvert);
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {



		// noslip BC 
		if(i==0 && user->bctype[0]==1) ucont[k][j][i].x = 0;
		if(i==mx-1 && user->bctype[1]==1) ucont[k][j][i-1].x = 0;
		if(j==0 && user->bctype[2]==1) ucont[k][j][i].y = 0;
		if(j==my-1 && user->bctype[3]==1) ucont[k][j-1][i].y = 0;
		if(k==0 && user->bctype[4]==1) ucont[k][j][i].z = 0;
		if(k==mz-1 && user->bctype[5]==1) ucont[k-1][j][i].z = 0;
		
		//cavity problem 
		if (j==my-1 && user->bctype[3]==2) ucont[k][j-1][i].y = 0;
		
		// couette flow j=0
		if (j==0 && user->bctype[2]==12) ucont[k][j][i].y = 0;
		
		// couette flow j=my-1
		if (j==my-1 && user->bctype[3]==12) ucont[k][j-1][i].y = 0;
		
		//slip BC
		if (user->bctype[0]==10 && i==0 && (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) ucont[k][j][i].x = 0;
		if (user->bctype[1]==10 && i==mx-1 && (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) ucont[k][j][i-1].x = 0;
		if (user->bctype[2]==10 && j==0 && (i!=0 && i!=mx-1 && k!=0 && k!=mz-1) ) ucont[k][j][i].y = 0;
		if (std::abs(user->bctype[3])==10 && j==my-1 && (i!=0 && i!=mx-1 && k!=0 && k!=mz-1) ) ucont[k][j-1][i].y = 0;
	}	
	DAVecRestoreArray(user->fda, user->Ucont, &ucont);
	DAVecRestoreArray(user->da, user->lNvert, &nvert);
	
	DAGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DAGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);

	Contra2Cart_2(user);
	IB_BC(user);

//        DAVecGetArray(user->da, user->lNvert, &nvert);
//	printf("*** the nvert %le \n", nvert[2][1][2]);
//        DAVecRestoreArray(user->da, user->lNvert, &nvert);
//  PetscInt rank;
//  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

//  if (rank == 0) printf ("******* Fromfunction_SNES was called \n");
	
	VecSet(Rhs,0);
	
	const double dt = user->dt;
	double coeff = time_coeff();
	if( coeff>0.9 && coeff<1.1 ) {
		VecAXPY(Rhs, -1./dt, user->Ucont);
		VecAXPY(Rhs, 1./dt, user->Ucont_o);
	}
	else/* if( coeff > 1.4 && coeff < 1.6 )*/ {
		VecAXPY(Rhs, -1.5/dt, user->Ucont);
		VecAXPY(Rhs, 2./dt, user->Ucont_o);
		VecAXPY(Rhs, -0.5/dt, user->Ucont_rm1);
	}/*
	else {
		VecAXPY(Rhs, -11./6./dt, user->Ucont);
		VecAXPY(Rhs, 3./dt, user->Ucont_o);
		VecAXPY(Rhs, -1.5/dt, user->Ucont_rm1);
		VecAXPY(Rhs, 1./3./dt, user->Ucont_rm2);
	}*/

        VecAXPY(Rhs, -1, user->dP); // xyang
	
	//Formfunction_2(user, Rhs, 1.0);	// careful ! adding values to Rhs
	if( coeff>0.9 && coeff<1.1 ) {
	  Formfunction_2(user, Rhs, 0.5);	// careful ! adding values to Rhs
	  VecAXPY(Rhs, 0.5, user->RHS_o);
	}
	else Formfunction_2(user, Rhs, 1.0);

        Vec Rhs_wm;
        VecDuplicate(Rhs, &Rhs_wm);
	VecSet(Rhs_wm,0.0);

	if (!ApproxBC_wm) {
	if ( (imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || IB_wm != 0)) {
		Formfunction_wm(user, Rhs_wm, 1.0);
		VecAXPY(Rhs, 1, Rhs_wm);
	}
	}

//	VecAXPY(Rhs, -1, user->dP); // xyang  move to up

//        abort();

/*

        if (IB_delta == 2) VecSet(user->lF_eul,0.0);

        if (IB_delta == 2) {

//          Pre_process(user, ibm);
            	Calc_Flagrnslip_g(user, ibm, fsi);

          	Calc_Feul_g(user, ibm, fsi);

	}
*/

        if (IB_delta || rotor_model) VecAXPY(Rhs, 1, user->F_eul);  // xyang 12-7-2010

	if (ApproxBC_wm) {
        if ( (imin_wm != 0 || imax_wm != 0 || jmin_wm != 0 || jmax_wm != 0 || IB_wm != 0)) VecAXPY(Rhs, 1, user->Force_wm);  // xyang 0521
	}

	if (temperature) VecAXPY(Rhs, 1, user->FTmprt);

	VecDestroy(Rhs_wm);
/*
        int aaa;

        TECIOOut_rhs(user, user->Ucont_rm1);
        cout << "Older Ucont \n";
        cin >> aaa;


	TECIOOut_rhs(user, user->Ucont_o);
        cout << "Old Ucont \n";
        cin >> aaa;

	TECIOOut_rhs(user, Ucont);
        cout << "Current Ucont \n";
        cin >> aaa;
*/
	//if( time_coeff()>1.1 && time_coeff()<2.0 ) VecScale(Rhs, 1./1.5);
	return 0;
}

void Force_Tmprt(UserCtx *user)
{
	DA		da = user->da;
	DALocalInfo	info;
	PetscInt	xs, xe, ys, ye, zs, ze; // Local grid information
	PetscInt	mx, my, mz; // Dimensions in three directions
	PetscInt	i, j, k;
	double nu = 1./(user->ren * user->Pr);
	double Ri = user->Ri;

	
	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	
	PetscReal 	***tmprt;
	PetscReal	***nvert;

	Cmpnts		***ftmprt, ***csi, ***eta, ***zet;

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

	DAVecGetArray(da, user->lNvert, &nvert);
	DAVecGetArray(user->da, user->lTmprt, &tmprt);
	DAVecGetArray(user->fda, user->lFTmprt, &ftmprt);
  	DAVecGetArray(user->fda, user->lCsi,  &csi);
  	DAVecGetArray(user->fda, user->lEta,  &eta);
  	DAVecGetArray(user->fda, user->lZet,  &zet);

	PetscReal tmprt_xzAvg[my];

	for (j=0; j<my; j++) {
		tmprt_xzAvg[j] = 0.0;
	}

	
        for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
        for (i=lxs; i<lxe; i++) {       // pressure node; cell center

		tmprt_xzAvg[j]+=tmprt[k][j][i];

	}


  	PetscBarrier(PETSC_NULL);

  	PetscReal     u_sum;
    	for (j=0; j<my; j++) {
      	
		PetscGlobalSum(&(tmprt_xzAvg[j]), &u_sum, PETSC_COMM_WORLD);
      		tmprt_xzAvg[j] = u_sum/((double)(mx*mz));

//		printf("the avg temperature %d %le \n", j, tmprt_xzAvg[j]);
    	}



	double force_x, force_y, force_z;
	
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {	// pressure node; cell center

		force_x = 0.0;
		force_y = 0.0;
		force_z = 0.0;

//		if (user->bctype_tmprt[0] || user->bctype_tmprt[1]) force_x = Ri * (tmprt[k][j][i]-tmprt_xzAvg[j]); 
		if (user->bctype_tmprt[2] || user->bctype_tmprt[3]) force_y = Ri * tmprt[k][j][i]; 
//		if (user->bctype_tmprt[4] || user->bctype_tmprt[5]) force_z = Ri * (tmprt[k][j][i]-tmprt_xzAvg[j]); 

	      	ftmprt[k][j][i].x = force_x *  csi[k][j][i].x + force_y *  csi[k][j][i].y + force_z *  csi[k][j][i].z;
	      	ftmprt[k][j][i].y = force_x *  eta[k][j][i].x + force_y *  eta[k][j][i].y + force_z *  eta[k][j][i].z;
	      	ftmprt[k][j][i].z = force_x *  zet[k][j][i].x + force_y *  zet[k][j][i].y + force_z *  zet[k][j][i].z;

//		if (k==3 && i==3) printf("the force  %d %d %le %le %le %le\n", my-2, j, ftmprt[k][j][i].y, force_x, force_y, force_z);



                if( nvert[k][j][i]>0.1) {
                  	ftmprt[k][j][i].x = 0.0;
                	ftmprt[k][j][i].y = 0.0;
                	ftmprt[k][j][i].z = 0.0;

                }
                else if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 || j==my-2) {
                        ftmprt[k][j][i].x = 0.0;
                        ftmprt[k][j][i].y = 0.0;
                        ftmprt[k][j][i].z = 0.0;

                }

		if (user->bctype_tmprt[1] && i==mx-2) ftmprt[k][j][i].x = 0.0;
		if (user->bctype_tmprt[3] && j==my-2) ftmprt[k][j][i].y = 0.0;
		if (user->bctype_tmprt[5] && k==mz-2) ftmprt[k][j][i].z = 0.0;


	}
	
	DAVecRestoreArray(da, user->lNvert, &nvert);
	DAVecRestoreArray(user->da, user->lTmprt, &tmprt);
	DAVecRestoreArray(user->fda, user->lFTmprt, &ftmprt);

	DALocalToGlobal(user->fda, user->lFTmprt, INSERT_VALUES, user->FTmprt);

//	TECIOOut_rhs(user,  user->FTmprt);

        DAVecRestoreArray(user->fda, user->lCsi,  &csi);
        DAVecRestoreArray(user->fda, user->lEta,  &eta);
        DAVecRestoreArray(user->fda, user->lZet,  &zet);
	
};


void Tmprt_BC(UserCtx *user)
{
	DA		da = user->da, fda = user->fda;
	DALocalInfo	info;
	PetscInt	xs, xe, ys, ye, zs, ze; // Local grid information
	PetscInt	mx, my, mz; // Dimensions in three directions
	PetscInt	i, j, k, ibi;
	//double nu = 1./user->ren;
	PetscReal	***aj, ***distance, ***rho, ***mu;
	
	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	
	PetscReal ***tmprt;
	Cmpnts	***csi, ***eta, ***zet, ***ucat;
	PetscReal	***nvert, ***ustar;

        double Ri = user->Ri;

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

	DAVecGetArray(fda, user->lCsi, &csi);
	DAVecGetArray(fda, user->lEta, &eta);
	DAVecGetArray(fda, user->lZet, &zet);
	DAVecGetArray(fda, user->lUcat, &ucat);
	
	DAVecGetArray(da, user->lNvert, &nvert);
	DAVecGetArray(da, user->lAj, &aj);
	DAVecGetArray(da, user->lUstar, &ustar);
	
	DAVecGetArray(user->da, user->lTmprt, &tmprt);
	
	if(levelset) {
		DAVecGetArray(da, user->lDensity, &rho);
		DAVecGetArray(da, user->lMu, &mu);
	}


	
	// BC for K, Omega
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {	// pressure node; cell center
		double ren = user->ren;
		//if(levelset) ren = rho[k][j][i]/ mu[k][j][i];
		
		// from saved inflow file
//		if(inletprofile==100 && k==1) {
//			K_Omega[k-1][j][i] = user->komega_plane[j][i];
//			K_Omega[k-1][j][i].x = std::max ( K_Omega[k-1][j][i].x, 0. );
//			K_Omega[k-1][j][i].y = std::max ( K_Omega[k-1][j][i].y, 1.e-4);
//		}
//		else if ( user->bctype_tmprt[4]==0 && k==1 && nvert[k][j][i]<0.1) {	// inflow
//			tmprt[k-1][j][i] = tmprt[k][j][i];
//		}
		
		// NO energy flux
		if ( user->bctype_tmprt[0] == 0 && i==1 ) tmprt[k][j][i-1] = tmprt[k][j][i];
		if ( user->bctype_tmprt[1] == 0 && i==mx-2 ) tmprt[k][j][i+1] = tmprt[k][j][i];
		if ( user->bctype_tmprt[2] == 0 && j==1 ) tmprt[k][j-1][i] = tmprt[k][j][i];
		if ( user->bctype_tmprt[3] == 0 && j==my-2 ) tmprt[k][j+1][i] = tmprt[k][j][i];
		if ( user->bctype_tmprt[4] == 0 && k==1 ) tmprt[k-1][j][i] = tmprt[k][j][i];
		if ( user->bctype_tmprt[5] == 0 && k==mz-2 ) tmprt[k+1][j][i] = tmprt[k][j][i];

		// Constant wall temperature difference 

		double sign_heat;
                if ( abs(user->bctype_tmprt[0]) == 1 && i==1    ) {
			sign_heat = user->bctype_tmprt[0]/abs(user->bctype_tmprt[0]);
			tmprt[k][j][i-1] = 2.0*sign_heat*0.5-tmprt[k][j][i];
		}

                if ( abs(user->bctype_tmprt[1]) == 1 && i==mx-2 ) {
			sign_heat = user->bctype_tmprt[1]/abs(user->bctype_tmprt[1]);
			tmprt[k][j][i+1] = 2.0*sign_heat*0.5-tmprt[k][j][i];
		}

                if ( abs(user->bctype_tmprt[2]) == 1 && j==1    ) {
			sign_heat = user->bctype_tmprt[2]/abs(user->bctype_tmprt[2]);
			tmprt[k][j-1][i] = 2.0*sign_heat*0.5-tmprt[k][j][i];
		}

                if ( abs(user->bctype_tmprt[3]) == 1 && j==my-2 ) {
			sign_heat = user->bctype_tmprt[3]/abs(user->bctype_tmprt[3]);
			tmprt[k][j+1][i] = 2.0*sign_heat*0.5-tmprt[k][j][i];
		}

                if ( abs(user->bctype_tmprt[4]) == 1 && k==1    ) {
			sign_heat = user->bctype_tmprt[4]/abs(user->bctype_tmprt[4]);
			tmprt[k-1][j][i] = 2.0*sign_heat*0.5-tmprt[k][j][i];
		}

                if ( abs(user->bctype_tmprt[5]) == 1 && k==mz-2 ) {
			sign_heat = user->bctype_tmprt[5]/abs(user->bctype_tmprt[5]);
			tmprt[k+1][j][i] = 2.0*sign_heat*0.5-tmprt[k][j][i];
		}


		// Uniform heat flux

		if ( abs(user->bctype_tmprt[0]) == 2 && i==1 ) {
			sign_heat = user->bctype_tmprt[0]/abs(user->bctype_tmprt[0]);
	                double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
        	        double hx = 1.0/aj[k][j][i]/area;
			tmprt[k][j][i-1] = sign_heat*hx + tmprt[k][j][i];
		}

                if ( abs(user->bctype_tmprt[1]) == 2 && i==mx-2 ) {
			sign_heat = user->bctype_tmprt[1]/abs(user->bctype_tmprt[1]);
                        double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
                        double hx = 1.0/aj[k][j][i]/area;
                        tmprt[k][j][i+1] = sign_heat*hx + tmprt[k][j][i];
                }


                if ( abs(user->bctype_tmprt[2]) == 2 && j==1 ) {
			sign_heat = user->bctype_tmprt[2]/abs(user->bctype_tmprt[2]);
                        double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
                        double hx = 1.0/aj[k][j][i]/area;
                        tmprt[k][j-1][i] = sign_heat*hx + tmprt[k][j][i];
                }

                if ( abs(user->bctype_tmprt[3]) == 2 && j==my-2 ) {
			sign_heat = user->bctype_tmprt[3]/abs(user->bctype_tmprt[3]);
                        double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
                        double hx = 1.0/aj[k][j][i]/area;
                        tmprt[k][j+1][i] = sign_heat*hx + tmprt[k][j][i];
                }


                if ( abs(user->bctype_tmprt[4]) == 2 && k==1 ) {
			sign_heat = user->bctype_tmprt[4]/abs(user->bctype_tmprt[4]);
                        double area = sqrt( zet[k][j][i].x*zet[k][j][i].x + zet[k][j][i].y*zet[k][j][i].y + zet[k][j][i].z*zet[k][j][i].z );
                        double hx = 1.0/aj[k][j][i]/area;
                        tmprt[k-1][j][i] = sign_heat*hx + tmprt[k][j][i];
                }

                if ( abs(user->bctype_tmprt[5]) == 2 && k==mz-2 ) {
			sign_heat = user->bctype_tmprt[5]/abs(user->bctype_tmprt[5]);
                        double area = sqrt( zet[k][j][i].x*zet[k][j][i].x + zet[k][j][i].y*zet[k][j][i].y + zet[k][j][i].z*zet[k][j][i].z );
                        double hx = 1.0/aj[k][j][i]/area;
                        tmprt[k+1][j][i] = sign_heat*hx + tmprt[k][j][i];
                }




/*			
		
		// wall function
		if( nvert[k][j][i]<1.1 && ( ( (user->bctype[0]==-1 || user->bctype[0]==-2) && i==1) || ( (user->bctype[1]==-1 || user->bctype[1]==-2) &&  i==mx-2) ) && (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) {
		  
		}
		
		if( nvert[k][j][i]<1.1 && ( ( (user->bctype[2]==-1 || user->bctype[2]==-2) && j==1) || ( (user->bctype[3]==-1 || user->bctype[3]==-2) &&  j==my-2) ) && (i!=0 && i!=mx-1 && k!=0 && k!=mz-1)) {
		  
		
		if ( nvert[k][j][i] > 1.1 ) K_Omega[k][j][i].x = K_Omega[k][j][i].y = 0.;
*/

	}
	DAVecRestoreArray(user->da, user->lTmprt, &tmprt);

        DALocalToLocalBegin(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);
        DALocalToLocalEnd(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);

        DAVecGetArray(user->da, user->lTmprt, &tmprt);
	
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
	
		if(flag) tmprt[k][j][i] = tmprt[c][b][a];
	}

/*	
	if(immersed)
	for(ibi=0; ibi<NumberOfBodies; ibi++)
	{
		extern IBMNodes *ibm_ptr;
		IBMNodes *ibm = &ibm_ptr[ibi];
		
		IBMListNode *current;
		current = user->ibmlist[ibi].head;
		while (current) {
			IBMInfo *ibminfo = &current->ibm_intp;
			//int ni = ibminfo->cell;
			current = current->next;
			double sb = ibminfo->d_s, sc = sb + ibminfo->d_i;
			int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
			int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
			int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
			i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
			double sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
			double Kc = (K_Omega[kp1][jp1][ip1].x * sk1 + K_Omega[kp2][jp2][ip2].x * sk2 + K_Omega[kp3][jp3][ip3].x * sk3);
			
			double ren = user->ren;
			if(levelset) ren = rho[k][j][i]/ mu[k][j][i];
				
			if(wallfunction && ti>0) {
				double utau = ustar[k][j][i];
				const double yplus_min = 0.25;
				sb = PetscMax( sb, 1./ren * yplus_min / utau );	// prevent the case sb=0
				double K, Ob;//, Om;
				
				// Wilcox pp.108-109
				K = utau*utau/sqrt(0.09);
				Ob = utau/sqrt(0.09)/(kappa*sb);
				
				K_Omega[k][j][i].x = K;
				K_Omega[k][j][i].y = Ob;
			}
			else {
				const double yplus_min = 0.25;
				
				double utau = ustar[k][j][i];
				
				K_Omega[k][j][i].x = Kc * sb / sc;
				sb = PetscMax( sb, 1./ren * yplus_min / utau );	// prevent the case sb=0
				K_Omega[k][j][i].y = wall_omega(ren, sb);	
				
				if ( K_Omega[k][j][i].x < 0 ) K_Omega[k][j][i].x = utau*utau/sqrt(0.09);
			}
			if(user->bctype[4]==5 && k==1) K_Omega[k][j][i]=K_Omega[k-1][j][i];
		};
	}
*/

	
	DAVecRestoreArray(fda, user->lCsi, &csi);
	DAVecRestoreArray(fda, user->lEta, &eta);
	DAVecRestoreArray(fda, user->lZet, &zet);
	DAVecRestoreArray(fda, user->lUcat, &ucat);
	
	DAVecRestoreArray(da, user->lNvert, &nvert);
	DAVecRestoreArray(da, user->lAj, &aj);
	DAVecRestoreArray(da, user->lUstar, &ustar);
	
	DAVecRestoreArray(user->da, user->lTmprt, &tmprt);
	
	if(levelset) {
		DAVecRestoreArray(da, user->lDensity, &rho);
		DAVecRestoreArray(da, user->lMu, &mu);
	}
	
	DALocalToLocalBegin(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);
	DALocalToLocalEnd(user->da, user->lTmprt, INSERT_VALUES, user->lTmprt);
};

