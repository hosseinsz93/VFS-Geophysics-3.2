// llevel for periodicity !!
#include "variables.h"

double dtau_levelset;
/*
extern double M(double a, double b);
extern int immersed, NumberOfBodies;
extern int i_periodic, j_periodic, k_periodic;
*/

/*
double density_viscosity(double &aj, Cmpnts &csi, Cmpnts &eta, &Cmpnts zet, double &level, double *rho, double *mu)
{
	if(levelset==1) {
		double dx = levelset_thickness ( aj, csi, eta, zet );
		*rho = rho_air + (rho_water - rho_air) * H ( level, dx );
		*mu = mu_air + (mu_water - mu_air) * H ( level, dx );
	}
	else if(levelset==2) {
		*rho = rho_air + (rho_water - rho_air) * level;
		*mu = mu_air + (mu_water - mu_air) * level;
	}
}
*/
void Init_Levelset_Vectors(UserCtx *user)
{
	VecDuplicate(user->P, &LevelSet);
	VecDuplicate(user->P, &LevelSet0);
	VecDuplicate(user->P, &LevelSet_o);
//	VecDuplicate(user->lP, &lLevelSet);
};

void Destroy_Levelset_Vectors(UserCtx *user)
{
	VecDestroy(&LevelSet);
	VecDestroy(&LevelSet0);
	VecDestroy(&LevelSet_o);
//	VecDestroy(&lLevelSet);
};

void Initialize_free_surface_location_vector(UserCtx *user)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	mx, my, mz;
	
	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	
	int xs = info.xs, xe = xs + info.xm;
	int ys = info.ys, ye = ys + info.ym;
	int zs = info.zs, ze = zs + info.zm;

	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	if(!my_rank) {
		user->free_surface_location = (double **) malloc( sizeof(double *) * mx );
		for(int i=0; i<mx; i++) {
			user->free_surface_location[i] = (double *) malloc( sizeof(double) * mz );
			for(int k=0; k<mz; k++) user->free_surface_location[i][k]=0;
		}
	}
};

void Calc_free_surface_location(UserCtx *user)
{
	DM da = user->da, fda = user->fda;
	DMDALocalInfo info;
	int mx, my, mz;
	PetscReal ***level, ***nvert;
	Cmpnts ***cent;
	
	DMDAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	
	int xs = info.xs, xe = xs + info.xm;
	int ys = info.ys, ye = ys + info.ym;
	int zs = info.zs, ze = zs + info.zm;

	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	double **buffer;
	
	//temporarily allocate
	buffer = (double **) malloc( sizeof(double *) * mx );
	for(int i=0; i<mx; i++) {
		buffer[i] = (double *) malloc( sizeof(double) * mz );
		for(int k=0; k<mz; k++) buffer[i][k]=-1000;
	}
	
	
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(fda, user->lCent, &cent);
	
	for (int k=lzs; k<lze; k++)
	for (int j=lys; j<lye; j++)
	for (int i=lxs; i<lxe; i++) {
		if(nvert[k][j][i]>0.1) continue;

		if( level[k][j][i]>=0 && level[k][j+1][i]<0 ) {	// water surface is above my cell center
			buffer[i][k] = cent[k][j][i].z + level[k][j][i];
		}
		else if( level[k][j][i]<0 && level[k][j-1][i]>=0 ) {	// water surface is below my cell center
			buffer[i][k] = cent[k][j][i].z + level[k][j][i];
		}
	}
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(fda, user->lCent, &cent);
	
	MPI_Reduce ( &buffer[0][0], &user->free_surface_location[0][0], mx*mz, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
	
	if(!my_rank) {
		for(int i=0; i<mx; i++)
		for(int k=0; k<mz; k++) if( (int) (user->free_surface_location[i][k])==-100 ) user->free_surface_location[i][k] = 0;
	}
	
	// free memeory
	for(int i=0; i<mx; i++) free ( buffer[i] );
	free ( buffer );
	
};


double dist(Cmpnts &a, Cmpnts &b)
{
  return sqrt( pow(a.x-b.x,2.) + pow(a.y-b.y,2.) + pow(a.z-b.z,2.) );
};


void Levelset_BC(UserCtx *user)
{
	DM		da = user->da;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***nvert, ***level, ***llevel, ***aj;
	Cmpnts ***cent;

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
	
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	DMDAVecGetArray(user->da, user->Levelset, &level);
	DMDAVecGetArray(user->da, user->lLevelset, &llevel);
	DMDAVecGetArray(user->fda, user->lCent, &cent);
	DMDAVecGetArray(user->da, user->lAj, &aj);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(levelset==2) {
			if(level[k][j][i]>1.0) level[k][j][i]=1.0;
			if(level[k][j][i]<0.0) level[k][j][i]=0.0;
		}
		
		if((int)(nvert[k][j][i]+0.1)==1) {
			int count=0;
			double sum=0;
			
			if (/*i<mx-3 &&*/ nvert[k][j][i+1]<0.1) {
			  /*
			  double AB = dist(cent[k][j][i], cent[k][j][i+1]);
			  double AC = dist(cent[k][j][i], cent[k][j][i+2]);
			  double lB = llevel[k][j][i+1];
			  double lC = llevel[k][j][i+2];
			  double val = lC - AC/AB * (lC - lB);
			 */ 
			  //if(nvert[k][j][i+2]<0.1) sum += val, count++;
			  sum += llevel[k][j][i+1], count++;
			}
			if (/*i>2 && */nvert[k][j][i-1]<0.1) {
			  /*
			  double AB = dist(cent[k][j][i], cent[k][j][i-1]);
                          double AC = dist(cent[k][j][i], cent[k][j][i-2]);
                          double lB = level[k][j][i-1];
                          double lC = level[k][j][i-2];
                          double val = lC - AC/AB * (lC - lB);
			  */
			  //if(nvert[k][j][i-2]<0.1) sum += val, count++;
			  sum += llevel[k][j][i-1], count++;
			}
			//
			if (/*j<my-3 && */nvert[k][j+1][i]<0.1) {
			  /*
			  double AB = dist(cent[k][j][i], cent[k][j+1][i]);
                          double AC = dist(cent[k][j][i], cent[k][j+2][i]);
                          double lB = llevel[k][j+1][i];
                          double lC = llevel[k][j+2][i];
                          double val = lC - AC/AB * (lC - lB);
			  */
			  //if(nvert[k][j+2][i]<0.1) sum += val, count++;
			  sum += llevel[k][j+1][i], count++;
			}
			if (/*j>2 && */nvert[k][j-1][i]<0.1) {
			  /*
			  double AB = dist(cent[k][j][i], cent[k][j-1][i]);
                          double AC = dist(cent[k][j][i], cent[k][j-2][i]);
                          double lB = llevel[k][j-1][i];
                          double lC = llevel[k][j-2][i];
                          double val = lC - AC/AB * (lC - lB);
			  */
			  //if(nvert[k][j-2][i]<0.1) sum += val, count++;
			  sum += llevel[k][j-1][i], count++;
			}
			//
			if (/*k<mz-3 &&*/ nvert[k+1][j][i]<0.1) {
			  /*
			  double AB = dist(cent[k][j][i], cent[k+1][j][i]);
                          double AC = dist(cent[k][j][i], cent[k+2][j][i]);
                          double lB = llevel[k+1][j][i];
                          double lC = llevel[k+2][j][i];
                          double val = lC - AC/AB * (lC - lB);
			  */
			  // if(nvert[k+2][j][i]<0.1) sum += val, count++;
			  sum += llevel[k+1][j][i], count++;
			}
			if (k>2 && nvert[k-1][j][i]<0.1) {
			  /*
			  double AB = dist(cent[k][j][i], cent[k-1][j][i]);
                          double AC = dist(cent[k][j][i], cent[k-2][j][i]);
                          double lB = llevel[k-1][j][i];
                          double lC = llevel[k-2][j][i];
                          double val = lC - AC/AB * (lC - lB);
			  */
			  //if(nvert[k-2][j][i]<0.1) sum += val, count++;
			  sum += llevel[k-1][j][i], count++;
			}
			
			if(count) {	// prevent NaN
				level[k][j][i] = sum / (double)count;  // xiaolei deactivate
			}
		}
	}

	// xiaolei change lzs-->zs  etc. for applying BCs. 
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				level[k][j][to] = llevel[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				level[k][j][to] = llevel[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				level[k][to][i] = llevel[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				level[k][to][i] = llevel[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
						
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(fix_inlet && levelset==1) {
					double y=cent[k][j][i].y, z=cent[k][j][i].z;
					if(inlet_y_flag) {
						level[k][j][i] = (inlet_y - y);
						level[to][j][i] = (inlet_y - y);
					}
					else if(inlet_z_flag) {
						level[k][j][i] = (inlet_z - z);
						level[to][j][i] = (inlet_z - z);
					}
				} else if(inflow_levelset && inletprofile==100 ) { // from saved inflow file   xiaolei 
					level[0][j][i] = user->level_plane[j][i];
					//level[1][j][i] = user->level_plane[j][i];
				} else {
					if(k_periodic) from = mz-2;
					else if(kk_periodic) from = -2;
					level[to][j][i] = llevel[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(fix_outlet && levelset==1) {
					double y=cent[k][j][i].y, z=cent[k][j][i].z;
					if(inlet_y_flag) {
						level[k][j][i] = (outlet_y - y);
						level[to][j][i] = (outlet_y - y);
					}
					else if(inlet_z_flag) {
						level[k][j][i] = (outlet_z - z);
						level[to][j][i] = (outlet_z - z);
					}
				}
				else {
					if(k_periodic) from = 1;
					else if(kk_periodic) from = mz+1;
					level[to][j][i] = llevel[from][j][i];
				}
			}
		}
	}
	
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
	DMDAVecRestoreArray(user->da, user->Levelset, &level);
	DMDAVecRestoreArray(user->da, user->lLevelset, &llevel);
	DMDAVecRestoreArray(user->fda, user->lCent, &cent);
	DMDAVecRestoreArray(user->da, user->lAj, &aj);
	
	DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
}

void Levelset_Function_IC(UserCtx *user)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	Cmpnts	***csi, ***eta, ***zet, ***cent;
	PetscReal	***nvert, ***level,***llevel,  ***aj, ***p;

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
	DMDAVecGetArray(da, user->Levelset, &level);
	DMDAVecGetArray(da, user->lLevelset, &llevel);
		
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(fda, user->lCent, &cent);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->P, &p);
		
	std::vector<double> elevation_k ( mz );	// for inlet-outlet interpolation
	if( (user->bctype[4]==5 || user->bctype[4]==4) || k_periodic || kk_periodic ) {
		if(inlet_y_flag) {
			elevation_k[0] = inlet_y;
			elevation_k[mz-1] = outlet_y;
			
			for (k=1; k<=mz-2; k++) {
				elevation_k[k] = ( outlet_y - inlet_y ) / (double) (mz -2 - 1) * (double) ( k - 1 ) + inlet_y;
			}
		}
		else if(inlet_z_flag) {
			elevation_k[0] = inlet_z;
			elevation_k[mz-1] = outlet_z;
			
			for (k=1; k<=mz-2; k++) {
				elevation_k[k] = ( outlet_z - inlet_z ) / (double) (mz -2 - 1) * (double) ( k - 1 ) + inlet_z;
			}
		}
		
	}
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double x0, y0, z0, R=0.15;
		double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;
		
		if( (user->bctype[4]==5 || user->bctype[4]==4) || k_periodic || kk_periodic ) {
			if(inlet_y_flag) level[k][j][i] = elevation_k[k] - y;
			else if(inlet_z_flag) level[k][j][i] = elevation_k[k] - z;
			p[k][j][i] = 0;
			if(level[k][j][i]>0 && nvert[k][j][i]<0.1) {
				if(inlet_y_flag)  p[k][j][i] = - rho_water * gravity_y * level[k][j][i];
				else if(inlet_z_flag) p[k][j][i] = - rho_water * gravity_z * level[k][j][i]; 
			}
		}
		else {
	
			// water drop
			/*
			x0=0.5, y0=0.5, z0=1.8;
			level[k][j][i] = R - sqrt ( pow( x0 - x, 2.0 ) + pow( y0 - y, 2.0 ) + pow( z0 - z, 2.0 ) );	//sphere
			if ( z < 1.2 ) level[k][j][i] = 1 - z;
			*/
			/*
			// bubble
			x0=0.5, y0=0.5, z0=0.20;
			level[k][j][i] = - R + sqrt ( pow( x0 - x, 2.0 ) + pow( y0 - y, 2.0 ) + pow( z0 - z, 2.0 ) );
			if( z > 0.8 ) level[k][j][i] = 1.0 - z;
			*/
			
			// sloshing, Fr should be 1.0
			if(sloshing==1) // 1d sloshing
			{
				double a=sloshing_a, b=sloshing_b, d=sloshing_d;	// a=0.05 (nonlinear), a=0.001 (linear)
				double k2 = 2*M_PI/b;
				double xi = d + a * cos ( 2.0 * M_PI * x / b );
				if(!inviscid) xi = d + a * cos ( k2*x );
				level[k][j][i] = xi - z;
			}
			else if(sloshing==2) { // 2d sloshing
				double L = 20., d=1., Beta=0.25, a=0.1; // L = width of the 3D tank
				double eta0 = a * exp ( -Beta * ( pow(x-L/2, 2) + pow(y-L/2, 2) ) );
				level[k][j][i] = eta0 + d - z;
			}
			else if(sloshing==3) { //solid body rotation
				level[k][j][i] = 1 - z;
			}
			
			if(bubble==1) {
				x0=0, y0=0, z0=bubble_z;
				R=bubble_d*0.5;
				
				level[k][j][i] = bubble_ws - z;
				level[k][j][i] = PetscMin(level[k][j][i],- R + sqrt ( pow( x0 - x, 2.0 ) + pow( y0 - y, 2.0 ) + pow( z0 - z, 2.0 ) ) );
				p[k][j][i] = 0;
				/*
				if(level[k][j][i]<0) p[k][j][i] = 0;
				else p[k][j][i] = - rho_water * gravity_z * level[k][j][i];
				if(z>bubble_ws) p[k][j][i] = 0;
				*/
			}
				
			if(inletprofile==20) {	// Enright test
				x0=0.35, y0=0.35, z0=0.35, R=0.15;
				level[k][j][i] = - R + sqrt ( pow( x0 - x, 2.0 ) + pow( y0 - y, 2.0 ) + pow( z0 - z, 2.0 ) );
			}
		}
					
		if(levelset==2) {
			if ( level[k][j][i]>0 ) level[k][j][i] = 1.0;
			else if ( level[k][j][i]<0 ) level[k][j][i] = 0;
		}
		
		if( nvert[k][j][i]>1.1 || i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ) {
			if(levelset==1) level[k][j][i] = -1;
		}
	}
		
	// Neumann , periodic conditions
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				level[k][j][to] = llevel[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				level[k][j][to] = llevel[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				level[k][to][i] = llevel[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				level[k][to][i] = llevel[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;
				
				if(user->bctype[4]==5) {
					if(inlet_y_flag) level[to][j][i] = 2.*(inlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(inlet_z - z) - level[k][j][i];
				} else if(inflow_levelset && inletprofile==100 ) { // from saved inflow file   xiaolei 
					level[0][j][i] = user->level_plane[j][i];
					//level[1][j][i] = user->level_plane[j][i];
				} 
				else {
					if(k_periodic) from = mz-2;
					else if(kk_periodic) from = -2;
					level[to][j][i] = llevel[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;

				if(/*user->bctype[5]==4 &&*/ fix_outlet) {
					/*if(inlet_y_flag) level[k][j][i] = outlet_y - y;
					else if(inlet_z_flag) level[k][j][i] = outlet_z - z;
					*/
					if(inlet_y_flag) level[to][j][i] = 2.*(outlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(outlet_z - z) - level[k][j][i];
				}
				else {
					if(k_periodic) from = 1;
					else if(kk_periodic) from = mz+1;
					level[to][j][i] = llevel[from][j][i];
				}
			}
		}
	}
	
	
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->Levelset, &level);
	DMDAVecRestoreArray(da, user->lLevelset, &llevel);
		
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(fda, user->lCent, &cent);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->P, &p);
	
	DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
        DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);

	DMGlobalToLocalBegin(user->da, user->P, INSERT_VALUES, user->lP);
	DMGlobalToLocalEnd(user->da, user->P, INSERT_VALUES, user->lP);
};

PetscErrorCode FormFunction_Levelset (SNES snes, Vec L, Vec Rhs, void *ptr)
{
	UserCtx *user = (UserCtx*)ptr;

	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal ***level;
	Cmpnts ***cent;

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
	
	DMDAVecGetArray(user->fda, user->lCent, &cent);
	
	
	DMGlobalToLocalBegin(user->da, L, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, L, INSERT_VALUES, user->lLevelset);
	
	DMDAVecGetArray(user->da, user->lLevelset, &level);
		
		
	// Neumann , periodic conditions
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				level[k][j][to] = level[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				level[k][j][to] = level[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				level[k][to][i] = level[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				level[k][to][i] = level[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;
				
				if(/*user->bctype[4]==5*/fix_inlet) {
					if(inlet_y_flag) level[to][j][i] = 2.*(inlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(inlet_z - z) - level[k][j][i];
				} else if(inflow_levelset && inletprofile==100 ) { // from saved inflow file   xiaolei 
					level[0][j][i] = user->level_plane[j][i];
				}
				else {
					if(k_periodic) from = mz-2;
					else if(kk_periodic) from = -2;
					level[to][j][i] = level[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;

				if(/*user->bctype[5]==4 &&*/ fix_outlet) {
					/*if(inlet_y_flag) level[k][j][i] = outlet_y - y;
					else if(inlet_z_flag) level[k][j][i] = outlet_z - z;
					*/
					if(inlet_y_flag) level[to][j][i] = 2.*(outlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(outlet_z - z) - level[k][j][i];
				}
				else {
					if(k_periodic) from = 1;
					else if(kk_periodic) from = mz+1;
					level[to][j][i] = level[from][j][i];
				}
			}
		}
	}
	
	DMDAVecRestoreArray(user->fda, user->lCent, &cent);
	DMDAVecRestoreArray(user->da, user->lLevelset, &level);
	
	Distance_Function_RHS(user, Rhs, 0);
	VecAXPY(Rhs, -1./dtau_levelset, L);
	VecAXPY(Rhs, 1./dtau_levelset, LevelSet_o);
	
	return(0);
}

void Solve_Reinit_explicit(UserCtx *user, int iter)
{
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal ***level;
	Cmpnts ***cent;

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
	
	DMDAVecGetArray(user->fda, user->lCent, &cent);
	
	DMGlobalToLocalBegin(user->da, LevelSet, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, LevelSet, INSERT_VALUES, user->lLevelset);
	
	DMDAVecGetArray(user->da, user->lLevelset, &level);
		
		
	// Neumann , periodic conditions
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				level[k][j][to] = level[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				level[k][j][to] = level[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				level[k][to][i] = level[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				level[k][to][i] = level[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;
				
				if(levelset==1 && fix_inlet) {
					if(inlet_y_flag) level[to][j][i] = 2.*(inlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(inlet_z - z) - level[k][j][i];
				} else if(inflow_levelset && inletprofile==100 ) { // from saved inflow file   xiaolei 
					level[0][j][i] = user->level_plane[j][i];
				}
				else {
					if(k_periodic) from = mz-2;
					else if(kk_periodic) from = -2;
					level[to][j][i] = level[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;

				if(levelset==1 && fix_outlet) {
					if(inlet_y_flag) level[to][j][i] = 2.*(outlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(outlet_z - z) - level[k][j][i];
				}
				else {
					if(k_periodic) from = 1;
					else if(kk_periodic) from = mz+1;
					level[to][j][i] = level[from][j][i];
				}
			}
		}
	}
	
	DMDAVecRestoreArray(user->fda, user->lCent, &cent);
	DMDAVecRestoreArray(user->da, user->lLevelset, &level);
	
	Vec Rhs;
	
	VecDuplicate(user->P, &Rhs);
		
	Distance_Function_RHS(user, Rhs, 0);
	
	VecAXPY(LevelSet, dtau_levelset, Rhs);
	VecDestroy(&Rhs);
	//VecAXPY(Rhs, -1./dtau_levelset, L);
	//VecAXPY(Rhs, 1./dtau_levelset, LevelSet_o);
	return;
}

void Solve_Reinit_implicit(UserCtx *user, int iter)
{
	SNES snes_distance;
	KSP ksp;
	PC pc;
	Vec r;
	Mat J;
	double norm;
	
	int bi=0;
	VecDuplicate(LevelSet, &r);
	SNESCreate(PETSC_COMM_WORLD,&snes_distance);
	SNESSetFunction(snes_distance,r,FormFunction_Levelset,(void *)&user[bi]);
	MatCreateSNESMF(snes_distance, &J);
	SNESSetJacobian(snes_distance,J,J,MatMFFDComputeJacobian,(void *)&user[bi]);
		
	
	SNESSetType(snes_distance, SNESTR);			//SNESTR,SNESLS	
	double tol=1.e-2;
	SNESSetMaxLinearSolveFailures(snes_distance,10000);
	SNESSetMaxNonlinearStepFailures(snes_distance,10000);		
	SNESKSPSetUseEW(snes_distance, PETSC_TRUE);
	SNESKSPSetParametersEW(snes_distance,3,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	SNESSetTolerances(snes_distance,PETSC_DEFAULT,tol,PETSC_DEFAULT,5,50000);
		
	SNESGetKSP(snes_distance, &ksp);
	KSPSetType(ksp, KSPGMRES);
	//KSPGMRESSetPreAllocateVectors(ksp);
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCNONE);
	
	int maxits=5;	//was 4
	double rtol=tol, atol=PETSC_DEFAULT, dtol=PETSC_DEFAULT;
	KSPSetTolerances(ksp,rtol,atol,dtol,maxits);
	
	SNESMonitorSet(snes_distance,MySNESMonitor,PETSC_NULL,PETSC_NULL);
	
	PetscPrintf(PETSC_COMM_WORLD, "\nSolving Levelset %d...\n", iter);
	SNESSolve(snes_distance, PETSC_NULL, LevelSet);
	VecCopy(LevelSet, user->Levelset);
	
	SNESGetFunctionNorm(snes_distance, &norm);
	PetscPrintf(PETSC_COMM_WORLD, "\nDistance SNES residual norm=%.5e\n\n", norm);
	VecDestroy(&r);
	MatDestroy(&J);
	SNESDestroy(&snes_distance);
};

void Reinit_Levelset(UserCtx *user)
{
	PetscReal ts,te,cput;
	PetscGetTime(&ts);
	
	int iter=0, maxit=levelset_it;
	
	//if(ti%50==1) maxit=std::max(maxit, 50);
	//if(ti<tistart+3) maxit=15;
	
	Init_Levelset_Vectors(user);
	
	Vec D;
	
	VecCopy (user->Levelset, LevelSet0);
	VecCopy (LevelSet0, LevelSet); 
	VecCopy (LevelSet0, LevelSet_o); 
	
	double norm, norm_old;
	
	dtau_levelset = dj_min * dtau_ratio;
	if(levelset==2) {
		double C=0.1;
		/*if(dthick_const>0) dtau_levelset = pow(dthick_const, 1.+d_levelset2()) * 0.05;
		  else */ dtau_levelset = pow(dj_min, 1.+d_levelset2()) * dtau_ratio;
		
		VecDuplicate(user->lUcat, &user->LevelsetGrad);
		Compute_LevelsetGrad0(user, user->LevelsetGrad);
	}
	
	VecDuplicate(LevelSet, &D);
	do {
		norm_old = norm;
		// Solve_Reinit_implicit(user, iter);
		Solve_Reinit_explicit(user, iter);
		VecWAXPY(D, -1., LevelSet_o, LevelSet);
		VecNorm(D, NORM_INFINITY, &norm);
			
		if(/*fabs(norm)<1.e-5 ||*/ iter++>= maxit ) break;
		VecCopy(LevelSet, LevelSet_o);
		
		//if(iter>500 && iter%500==0) dtau_levelset *= 2;
	} while(1);
	
	VecDestroy(&D);
	if(levelset==2) VecDestroy(&user->LevelsetGrad);
		
	DMDALocalInfo	info;
	int	i, j, k;
	DMDAGetLocalInfo(user->da, &info);
	int	mx = info.mx, my = info.my, mz = info.mz;
	int	xs = info.xs, xe = xs + info.xm;
	int	ys = info.ys, ye = ys + info.ym;
	int	zs = info.zs, ze = zs + info.zm;
	int	lxs = xs, lxe = xe;
	int	lys = ys, lye = ye;
	int	lzs = zs, lze = ze;
	PetscReal ***level, ***llevel, ***nvert;
	Cmpnts ***cent;
	
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	VecCopy(LevelSet, user->Levelset);
	DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	DMDAVecGetArray(user->da, user->Levelset, &level);
	DMDAVecGetArray(user->da, user->lLevelset, &llevel);
	DMDAVecGetArray(user->fda, user->lCent, &cent);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(nvert[k][j][i]>0.1) {
			if ( nvert[k][j][i+1]+nvert[k][j][i-1]+nvert[k+1][j][i]+nvert[k-1][j][i] > 3.9 ) continue;
			double count=0;
			double sum=0;
			
			if (nvert[k][j][i+1]>0.1) sum += llevel[k][j][i+1], count+=1.;
			if (nvert[k][j][i- 1]>0.1) sum += llevel[k][j][i -1], count+=1.;
			if (nvert[k+1][j][i]>0.1) sum += llevel[k+1][j][i], count+=1.;
			if (nvert[k- 1][j][i]>0.1) sum += llevel[k -1][j][i], count+=1.;
			
			level[k][j][i] = sum / count;
		}
	}
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				level[k][j][to] = llevel[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				level[k][j][to] = llevel[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				level[k][to][i] = llevel[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				level[k][to][i] = llevel[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(/*user->bctype[4]==5*/fix_inlet) {
					double y=cent[k][j][i].y, z=cent[k][j][i].z;
					if(inlet_y_flag) level[to][j][i] = 2.*(inlet_y - y) - llevel[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(inlet_z - z) - llevel[k][j][i];                           
				} else if(inflow_levelset && inletprofile==100 ) { // from saved inflow file   xiaolei 
					level[0][j][i] = user->level_plane[j][i];
				}
				else {
					if(k_periodic) from = mz-2;
					else if(kk_periodic) from = -2;
					level[to][j][i] = llevel[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(/*user->bctype[5]==4 &&*/ fix_outlet) {
				  double y=cent[k][j][i].y, z=cent[k][j][i].z;
				  if(inlet_y_flag) level[to][j][i] = 2.*(outlet_y - y) - llevel[k][j][i];
				  else if(inlet_z_flag) level[to][j][i] = 2.*(outlet_z - z) - llevel[k][j][i];
                                }
				else {
				  if(k_periodic) from = 1;
				  else if(kk_periodic) from = mz+1;
				  level[to][j][i] = llevel[from][j][i];
				}
			}
		}
	}
	
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
	DMDAVecRestoreArray(user->da, user->Levelset, &level);
	DMDAVecRestoreArray(user->da, user->lLevelset, &llevel);
	DMDAVecRestoreArray(user->fda, user->lCent, &cent);

	DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
        DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	
	Destroy_Levelset_Vectors(user);
	
	PetscGetTime(&te);
	cput=te-ts;
	if (!my_rank) {
		FILE *f;
		char filen[80];
  		sprintf(filen, "%s/Converge_dU", path);
		f = fopen(filen, "a");
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d(levelset) %.2e(s) %le\n", ti, cput, norm);
		fclose(f);
	}
};

void Compute_Density(UserCtx *user)
{
	DM		da = user->da;
	DM		fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***nvert, ***level, ***rho, ***mu, ***aj;
	Cmpnts ***csi, ***eta, ***zet;

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
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lDensity, &rho);
	DMDAVecGetArray(da, user->lMu, &mu);
	DMDAVecGetArray(da, user->lAj, &aj);
	
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);

	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		
		if(levelset==1) {
			double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
			rho[k][j][i] = rho_air + (rho_water - rho_air) * H ( level[k][j][i], dx );  // xiaolei H_1
			mu[k][j][i] = mu_air + (mu_water - mu_air) * H ( level[k][j][i], dx );  // xiaolei H_1

/*
			// xiaolei add
			if (nvert[k][j][i]>0.1 || nvert[k][j][i+1]>0.1 || nvert[k][j][i-1]>0.1 || nvert[k][j+1][i]>0.1 || nvert[k][j-1][i]>0.1 || nvert[k+1][j][i]>0.1 || nvert[k-1][j][i]>0.1) {
				if (level[k][j][i]<0.0) {
					rho[k][j][i] = rho_air;
					mu[k][j][i] = mu_air;
				} else
				{
					rho[k][j][i] = rho_water;
					mu[k][j][i] = mu_water;
				}	 
			}
			// end add
*/

		}
		else if(levelset==2) {
		  //if(level[k][j][i]>1.0) level[k][j][i]=1.0;
		  //if(level[k][j][i]<0.0) level[k][j][i]=0.0;
			
			rho[k][j][i] = rho_air + (rho_water - rho_air) * level[k][j][i];
			mu[k][j][i] = mu_air + (mu_water - mu_air) * level[k][j][i];
		}
		
		if(rho[k][j][i]<0) printf("Negative density ! rho=%f, phi=%f, (%d,%d,%d)\n", rho[k][j][i], level[k][j][i], i, j, k);
		if(mu[k][j][i]<0) printf("Negative viscosity! mu=%f, phi=%f\n, (%d,%d,%d)", mu[k][j][i], level[k][j][i], i, j, k);

		
		if(i==1) rho[k][j][i-1]=rho[k][j][i];
		if(i==mx-2) rho[k][j][i+1]=rho[k][j][i];
		if(j==1) rho[k][j-1][i]=rho[k][j][i];
		if(j==my-2) rho[k][j+1][i]=rho[k][j][i];
		if(k==1) rho[k-1][j][i]=rho[k][j][i];
		if(k==mz-2) rho[k+1][j][i]=rho[k][j][i];
		
		
		if(i==1) mu[k][j][i-1]=mu[k][j][i];
		if(i==mx-2) mu[k][j][i+1]=mu[k][j][i];
		if(j==1) mu[k][j-1][i]=mu[k][j][i];
		if(j==my-2) mu[k][j+1][i]=mu[k][j][i];
		if(k==1) mu[k-1][j][i]=mu[k][j][i];
		if(k==mz-2) mu[k+1][j][i]=mu[k][j][i];
		


		//if( nvert[k][j][i]>0.1) rho[k][j][i] = 0;
		//if( nvert[k][j][i]>0.1) mu[k][j][i] = 0;
	}
		
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lDensity, &rho);
	DMDAVecRestoreArray(da, user->lMu, &mu);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	
	DMDALocalToLocalBegin(da, user->lDensity, INSERT_VALUES, user->lDensity);
	DMDALocalToLocalEnd(da, user->lDensity, INSERT_VALUES, user->lDensity);
	
	DMDALocalToLocalBegin(da, user->lMu, INSERT_VALUES, user->lMu);
	DMDALocalToLocalEnd(da, user->lMu, INSERT_VALUES, user->lMu);
	
	if(ti==tistart) {
		DMDALocalToLocalBegin(da, user->lNvert, INSERT_VALUES, user->lNvert);
		DMDALocalToLocalEnd(da, user->lNvert, INSERT_VALUES, user->lNvert);
	}
	
	DMDAVecGetArray(da, user->lDensity, &rho);
	DMDAVecGetArray(da, user->lMu, &mu);

	// Neumann , periodic conditions

	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				rho[k][j][to] = rho[k][j][from];
				mu[k][j][to] = mu[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				rho[k][j][to] = rho[k][j][from];
				mu[k][j][to] = mu[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				rho[k][to][i] = rho[k][from][i];
				mu[k][to][i] = mu[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				rho[k][to][i] = rho[k][from][i];
				mu[k][to][i] = mu[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				rho[to][j][i] = rho[from][j][i];
				mu[to][j][i] = mu[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				rho[to][j][i] = rho[from][j][i];
				mu[to][j][i] = mu[from][j][i];
			}
		}
	}

	
	DMDAVecRestoreArray(da, user->lDensity, &rho);
	DMDAVecRestoreArray(da, user->lMu, &mu);

	/*
	// xiaolei add for test
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		rho[k][j][i] = 1;
		mu[k][j][i] = 0.00001;
	}
	*/

	//char fname[80];
	//sprintf(fname,"Density");
	//TECIOOut_rhs_da(user, user->lDensity, fname);

	//sprintf(fname,"Mu");
	//TECIOOut_rhs_da(user, user->lMu, fname);
	


};

void Compute_Water_Volume(UserCtx *user, double *vol)
{
	DM		da = user->da;
	DM		fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***nvert, ***level, ***rho, ***aj;
	Cmpnts ***csi, ***eta, ***zet;
	
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

	double lvol = 0, lw;
	
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lDensity, &rho);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);


	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(nvert[k][j][i]<0.1) {
			//double dx = pow(1./aj[k][j][i], 1./3.);
			/*
			double dx = 1./aj[k][j][i]/sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
			if(dthick_set) dx*=dthick;
			*/
			double vf;
			if(levelset==1) {
				double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
				vf = H(level[k][j][i], dx);
			}
			else if(levelset==2) {
				vf = level[k][j][i];
			}
						
			//double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
			//double vf = H(level[k][j][i], dx);
			lvol += vf / aj[k][j][i];
		}
	}
		
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lDensity, &rho);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	
	//GlobalSum_All(&lvol, vol, PETSC_COMM_WORLD);
	MPI_Allreduce(&lvol, vol, 1, MPIU_SCALAR,MPIU_SUM, PETSC_COMM_WORLD);
};

void Correct_Volume_Flux_Levelset(UserCtx *user)
{
	if(user->bctype[3]!=-10) return;
	
	DM		da = user->da;
	DM		fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***nvert, ***level, ***rho, ***aj;
	Cmpnts ***ucont, ***ucat, ***eta;

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
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lDensity, &rho);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(fda, user->Ucont, &ucont);
	DMDAVecGetArray(fda, user->Ucat, &ucat);
	DMDAVecGetArray(fda, user->lEta, &eta);
	
	double lArea = 0, lJFlux = 0;
	
	if(ye==my && user->bctype[3]==-10){
		j=my-2;
		for(k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if (nvert[k][j][i] < 0.1) {
				double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				lArea += area;
				ucont[k][j][i].y = 0.5*(ucat[k][j][i].x+ucat[k][j+1][i].x)*eta[k][j][i].x + 0.5*(ucat[k][j][i].y+ucat[k][j+1][i].y)*eta[k][j][i].y + 0.5*(ucat[k][j][i].z+ucat[k][j+1][i].z)*eta[k][j][i].z;;
				ucat[k][j+1][i] = ucat[k][j][i];
				//ucont[k][j][i].y = 0;
				lJFlux += ucont[k][j][i].y;
			}
		}
	}

	double SumJFlux, SumArea;
	
	GlobalSum_All(&lArea, &SumArea, PETSC_COMM_WORLD);
	GlobalSum_All(&lJFlux, &SumJFlux, PETSC_COMM_WORLD);
	
	double Flux = -SumJFlux + (water_vol - water_vol_o) / user->dt;
	
	if(ye==my){
		j=my-2;
		for(k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if (nvert[k][j][i] < 0.1) {
				double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				//ucont[k][j][i].y += Flux * area / SumArea;
				ucat[k][j+1][i] = ucat[k][j][i];
			}
		}
	}
	
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lDensity, &rho);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(fda, user->Ucont, &ucont);
	DMDAVecRestoreArray(fda, user->Ucat, &ucat);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	
	DMGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	
	DMGlobalToLocalBegin(user->fda, user->Ucat, INSERT_VALUES, user->lUcat);
	DMGlobalToLocalEnd(user->fda, user->Ucat, INSERT_VALUES, user->lUcat);
}



void Levelset_Advect_RHS(UserCtx *user, Vec DRHS)
{
	DM 		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs, xe, ys, ye, zs, ze;
	int	mx, my, mz;
	int	i, j, k;
	
	Vec	Aj  = user->lAj;

	Cmpnts	***ucont, ***kzet, ***cent;
	PetscReal	***aj;
	PetscReal	***level, ***rhs, ***nvert;
	int	lxs, lys, lzs, lxe, lye, lze;

	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  	mx = info.mx; my = info.my; mz = info.mz;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  
	VecSet(DRHS,0);
	
	DMDAVecGetArray(fda, user->lCent, &cent);
	
	DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
		
	DMDAVecGetArray(user->da, user->lLevelset, &level);
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				level[k][j][to] = level[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				level[k][j][to] = level[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				level[k][to][i] = level[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				level[k][to][i] = level[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;
				
				if(fix_inlet && levelset==1) {
					if(inlet_y_flag) level[to][j][i] = 2.*(inlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(inlet_z - z) - level[k][j][i];
				} else if(inflow_levelset && inletprofile==100 ) { // from saved inflow file   xiaolei 
					level[0][j][i] = user->level_plane[j][i];
				}	
				else {
					if(k_periodic) from = mz-2;
					else if(kk_periodic) from = -2;
					level[to][j][i] = level[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				double x = cent[k][j][i].x, y = cent[k][j][i].y, z = cent[k][j][i].z;
				
				if(fix_outlet && levelset==1) {
					if(inlet_y_flag) level[to][j][i] = 2.*(outlet_y - y) - level[k][j][i];
					else if(inlet_z_flag) level[to][j][i] = 2.*(outlet_z - z) - level[k][j][i];
				}
				else {
					if(k_periodic) from = 1;
					else if(kk_periodic) from = mz+1;
					level[to][j][i] = level[from][j][i];
				}
			}
		}
	}
	
	DMDAVecRestoreArray(user->da, user->lLevelset, &level);
  
	DMDAVecGetArray(da,  Aj,  &aj);
	DMDAVecGetArray(fda,  user->lKZet,  &kzet);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(fda, user->lUcont, &ucont);
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, DRHS, &rhs);
  
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double U_dpdc=0, U_dpde=0, U_dpdz=0;
		
		if (nvert[k][j][i]>0.1) {
			rhs[k][j][i]=0.;
			continue;
		}
		
		if(levelset==2 && ti>0) {
			if(fix_inlet && k==1) continue;
			else if(fix_outlet && k==mz-2) continue;
		}
			
		double densityL, densityR;
		double aL=ucont[k][j][i-1].x, aR=ucont[k][j][i].x;	// wave speed
		/*
		if ( i==mx-2 && !i_periodic && !ii_periodic) {
			densityL = level[k][j][i-1] + 0.5*M(level[k][j][i]-level[k][j][i-1], level[k][j][i-1]-level[k][j][i-2]);
			
			if( !i_periodic && !ii_periodic ) densityR = level[k][j][i];
			else densityR = level[k][j][i] + 0.5*M(level[k][j][i+1]-level[k][j][i], level[k][j][i]-level[k][j][i-1]);
		}
		else if ( i==1 && !i_periodic && !ii_periodic ) {
			if( !i_periodic && !ii_periodic ) densityL = level[k][j][i];// - 0.5*M(level[k][j][i+1]-level[k][j][i], level[k][j][i]-level[k][j][i-1]);
			else densityL = level[k][j][i] - 0.5*M(level[k][j][i+1]-level[k][j][i], level[k][j][i]-level[k][j][i-1]);
			
			densityR = level[k][j][i+1] - 0.5*M(level[k][j][i+2]-level[k][j][i+1], level[k][j][i+1]-level[k][j][i]);
		}
		else*/ {
			int iR=i+1, iRR=i+2;
			int iL=i-1, iLL=i-2;
			
			if(i==1) {
				if(i_periodic) iLL = mx-3;
				else if(ii_periodic) iLL=-3;
				else iL=1, iLL=1;
			}
			else if(i==mx-2) {
				if(i_periodic) iRR = 2;
				else if(ii_periodic) iRR=mx+2;
				else iR=mx-2, iRR=mx-2;
			}
			
			//if(nvert[k][j][i-1]>0.1) iL = i;
			if(i>1 && nvert[k][j][i-2]>0.1) iLL = iL;
			//if(nvert[k][j][i+1]>0.1) iR = i;
			if(i<mx-2 && nvert[k][j][i+2]>0.1) iRR = iR;
			
			densityL = weno3(level[k][j][iLL],level[k][j][iL],level[k][j][i],level[k][j][iR],aL);
			densityR = weno3(level[k][j][iL],level[k][j][i],level[k][j][iR],level[k][j][iRR],aR);
		}
		U_dpdc = densityR*ucont[k][j][i].x - densityL*ucont[k][j][i-1].x;
		//U_dpdc = 0.5*(ucont[k][j][i].x+ucont[k][j][i-1].x) * (densityR - densityL);
		
		aL=ucont[k][j-1][i].y, aR=ucont[k][j][i].y;
		/*
		if ( j==my-2 && !j_periodic && !jj_periodic ) {
			densityL = level[k][j-1][i] + 0.5*M(level[k][j][i]-level[k][j-1][i], level[k][j-1][i]-level[k][j-2][i]);
			
			if( !j_periodic && !jj_periodic ) densityR = level[k][j][i];// + 0.5*M(level[k][j+1][i]-level[k][j][i], level[k][j][i]-level[k][j-1][i]);
			else densityR = level[k][j][i] + 0.5*M(level[k][j+1][i]-level[k][j][i], level[k][j][i]-level[k][j-1][i]);
		}
		else if ( j==1 && !j_periodic && !jj_periodic ) {
			if( !j_periodic && !jj_periodic ) densityL = level[k][j][i];// - 0.5*M(level[k][j+1][i]-level[k][j][i], level[k][j][i]-level[k][j-1][i]);
			else densityL = level[k][j][i] - 0.5*M(level[k][j+1][i]-level[k][j][i], level[k][j][i]-level[k][j-1][i]);
			
			densityR = level[k][j+1][i] - 0.5*M(level[k][j+2][i]-level[k][j+1][i], level[k][j+1][i]-level[k][j][i]);
		}
		else*/ {
			int jR=j+1, jRR=j+2;
			int jL=j-1, jLL=j-2;
			
			if(j==1) {
				if(j_periodic) jLL = my-3;
				else if(jj_periodic) jLL=-3;
				else jL=1, jLL=1;
			}
			else if(j==my-2) {
				if(j_periodic) jRR = 2;
				else if(jj_periodic) jRR=my+2;
				else jR=my-2, jRR=my-2;
			}
			
			//if(nvert[k][j-1][i]>0.1) jL = j;
			if(j>1 && nvert[k][j-2][i]>0.1) jLL = jL;
			//if(nvert[k][j+1][i]>0.1) jR = j;
			if(j<my-2 && nvert[k][j+2][i]>0.1) jRR = jR;
			
			densityL = weno3(level[k][jLL][i],level[k][jL][i],level[k][j][i],level[k][jR][i],aL);
			densityR = weno3(level[k][jL][i],level[k][j][i],level[k][jR][i],level[k][jRR][i],aR);
		}
		U_dpde = densityR*ucont[k][j][i].y - densityL*ucont[k][j-1][i].y;
		//U_dpde = 0.5*(ucont[k][j][i].y+ucont[k][j-1][i].y) * (densityR - densityL);
		
		aL=ucont[k-1][j][i].z, aR=ucont[k][j][i].z;
		/*
		if ( k==mz-2 && !k_periodic && !kk_periodic ) {
			densityL = level[k-1][j][i] + 0.5*M(level[k][j][i]-level[k-1][j][i], level[k-1][j][i]-level[k-2][j][i]);
			
			if( !k_periodic && !kk_periodic ) densityR = level[k][j][i];
			else densityR = level[k][j][i] + 0.5*M(level[k+1][j][i]-level[k][j][i], level[k][j][i]-level[k-1][j][i]);
		}
		else if ( k==1 && !k_periodic && !kk_periodic ) {
			if( !k_periodic && !kk_periodic ) densityL = level[k][j][i];
			else densityL = level[k][j][i] - 0.5*M(level[k+1][j][i]-level[k][j][i], level[k][j][i]-level[k-1][j][i]);
			
			densityR = level[k+1][j][i] - 0.5*M(level[k+2][j][i]-level[k+1][j][i], level[k+1][j][i]-level[k][j][i]);
		}
		else */ {
			int kR=k+1, kRR=k+2;
			int kL=k-1, kLL=k-2;
			
			if(k==1) {
				if(k_periodic) kLL = mz-3;
				else if(kk_periodic) kLL=-3;
				else kL=1, kLL=1;  
			}
			else if(k==mz-2) {
				if(k_periodic) kRR = 2;
				else if(kk_periodic) kRR=mz+2;
				else kR=mz-2, kRR=mz-2;
			}
			
			//if(nvert[k-1][j][i]>0.1) kL = k;
			if(k>1 && nvert[k-2][j][i]>0.1) kLL = kL;
			//if(nvert[k+1][j][i]>0.1) kR = k;
			if(k<mz-2 && nvert[k+2][j][i]>0.1) kRR = kR;
			
			densityL = weno3(level[kLL][j][i],level[kL][j][i],level[k][j][i],level[kR][j][i],aL);
			densityR = weno3(level[kL][j][i],level[k][j][i],level[kR][j][i],level[kRR][j][i],aR);
		}
		U_dpdz = densityR*ucont[k][j][i].z - densityL*ucont[k-1][j][i].z;
		//U_dpdz = 0.5*(ucont[k][j][i].z+ucont[k-1][j][i].z) * (densityR - densityL);
			
		rhs[k][j][i] = - ( U_dpdc + U_dpde + U_dpdz ) * aj[k][j][i];	// advection
	}
	
	DMDAVecRestoreArray(fda, user->lCent, &cent);
	DMDAVecRestoreArray(da,  Aj,  &aj);
	DMDAVecRestoreArray(fda,  user->lKZet,  &kzet);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(fda, user->lUcont, &ucont);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, DRHS, &rhs);

	if (sublevel && (les || rans)) Compute_Force_Subgridlevel(user, DRHS);  // xiaolei add.   
}

void Advect_Levelset(UserCtx *user, double dt)
{
	VecCopy(user->Levelset, user->Levelset_o);
	
	Vec R0, R1;
	VecDuplicate(user->Levelset, &R0);	// allocation
	VecDuplicate(user->Levelset, &R1);	// allocation
	
	Levelset_Advect_RHS(user, R0);
	VecAXPY(user->Levelset, dt, R0);        /* U(1) = U(n) + dt * RHS(n) */
	
	VecWAXPY(user->Levelset, dt, R0, user->Levelset_o);	
	Levelset_Advect_RHS(user, R1);
	VecWAXPY(user->Levelset, 0.5*dt, R0, user->Levelset_o);
	VecAXPY(user->Levelset, 0.5*dt, R1);
	
	VecDestroy(&R0);		// free
	VecDestroy(&R1);		// free
	
	DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	
	
}

void Compute_Surface_Tension(UserCtx *user)
{
	DM 		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs, xe, ys, ye, zs, ze;
	int	mx, my, mz;
	int	i, j, k;
	int	lxs, lys, lzs, lxe, lye, lze;

	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  	mx = info.mx; my = info.my; mz = info.mz;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	Cmpnts ***curv, ***stension;
	Cmpnts	***csi, ***eta, ***zet;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	PetscReal ***aj, ***iaj, ***jaj, ***kaj;
	PetscReal ***level, ***nvert, ***density, ***grad, ***h;
	
	DMDAVecGetArray(fda, user->Curv, &curv);
	DMDAVecGetArray(da, user->Grad_abs, &grad);
	DMDAVecGetArray(da, user->Heaviside, &h);
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lDensity, &density);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	
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
	
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);
	
	DMDAVecGetArray(fda, user->lST, &stension);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double phi = level[k][j][i];
		double ajc = aj[k][j][i];
		double kappa = (curv[k][j][i].x - curv[k][j][i-1].x + curv[k][j][i].y - curv[k][j-1][i].y + curv[k][j][i].z - curv[k-1][j][i].z) * ajc;
		//double dx=pow(1./aj[k][j][i],1./3.);
		/*
		double dx = 1./aj[k][j][i]/sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
		if(dthick_set) dx *= dthick;
		*/
		double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
		double gradH = dH (phi, dx); 
		double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
		double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
		double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
		double dldc, dlde, dldz, dl_dx, dl_dy, dl_dz;
		double dhdc, dhde, dhdz, dh_dx, dh_dy, dh_dz;
		
		double sigma = 7.28e-2;
		
		Compute_dscalar_center (i, j, k, mx, my, mz, level, nvert, &dldc, &dlde, &dldz );
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
		
		Compute_dscalar_center (i, j, k, mx, my, mz, h, nvert, &dhdc, &dhde, &dhdz );
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dhdc, dhde, dhdz, &dh_dx, &dh_dy, &dh_dz);
		
		double A = sigma * kappa / density[k][j][i];// * gradH;// / density[k][j][i];
		
		stension[k][j][i].x = - A * dl_dx * gradH;
		stension[k][j][i].y = - A * dl_dy * gradH;
		stension[k][j][i].z = - A * dl_dz * gradH;
		
		if ( nvert[k][j][i]> 0.1 ) Set (&stension[k][j][i], 0);

	}
	
	DMDAVecRestoreArray(fda, user->Curv, &curv);
	DMDAVecRestoreArray(da, user->Grad_abs, &grad);
	DMDAVecRestoreArray(da, user->Heaviside, &h);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lDensity, &density);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	
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
	
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	
	DMDAVecRestoreArray(fda, user->lST, &stension);
	
	DMDALocalToLocalBegin(fda, user->lST, INSERT_VALUES, user->lST);
	DMDALocalToLocalEnd(fda, user->lST, INSERT_VALUES, user->lST);
	
	DMDAVecGetArray(fda, user->lST, &stension);
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				stension[k][j][to] = stension[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				stension[k][j][to] = stension[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				stension[k][to][i] = stension[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				stension[k][to][i] = stension[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				stension[to][j][i] = stension[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				stension[to][j][i] = stension[from][j][i];
			}
		}
	}
	DMDAVecRestoreArray(fda, user->lST, &stension);
	
}



void Compute_Surface_Curv(UserCtx *user)
{
	DM 		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs, xe, ys, ye, zs, ze;
	int	mx, my, mz;
	int	i, j, k;
	int	lxs, lys, lzs, lxe, lye, lze;

	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  	mx = info.mx; my = info.my; mz = info.mz;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	Cmpnts ***curv;
	Cmpnts	***csi, ***eta, ***zet;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	PetscReal ***aj, ***iaj, ***jaj, ***kaj;
	PetscReal ***level, ***nvert, ***grad, ***h;
	
	DMDAVecGetArray(fda, user->Curv, &curv);
	DMDAVecGetArray(da, user->Grad_abs, &grad);
	DMDAVecGetArray(da, user->Heaviside, &h);
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	
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
	
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);
	
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double dldc, dlde, dldz;
		double dl_dx, dl_dy, dl_dz;
		double ajc = aj[k][j][i];
		double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
		double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
		double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
		
		Compute_dscalar_center (i, j, k, mx, my, mz, level, nvert, &dldc, &dlde, &dldz );
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
		grad[k][j][i] = sqrt ( dl_dx*dl_dx + dl_dy*dl_dy + dl_dz*dl_dz );
		//double dx=pow(1./aj[k][j][i],1./3.);
		/*
		double dx = 1./aj[k][j][i]/sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
		if(dthick_set) dx *= dthick;
		*/
		double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
		h[k][j][i] = H ( level[k][j][i], dx );
		/*
		int c=k, b=j, a=i, flag=0;
		
		// Neumann conditions
		if(i==1) a=0, flag=1;
		if(i==mx-2) a=mx-1, flag=1;
		if(j==1) b=0, flag=1;
		if(j==my-2) b=my-1, flag=1;
		if(k==1) c=0, flag=1;
		if(k==mz-2) c=mz-1, flag=1;
		
		if(flag) {
			grad[c][b][a] = grad[k][j][i];
			h[c][b][a] = h[k][j][i];
		}
		*/
	}
	
	DMDAVecRestoreArray(da, user->Grad_abs, &grad);
	DMDAVecRestoreArray(da, user->Heaviside, &h);
	
	DMDALocalToLocalBegin(da, user->Grad_abs, INSERT_VALUES, user->Grad_abs);
	DMDALocalToLocalEnd(da, user->Grad_abs, INSERT_VALUES, user->Grad_abs);
	
	DMDALocalToLocalBegin(da, user->Heaviside, INSERT_VALUES, user->Heaviside);
	DMDALocalToLocalEnd(da, user->Heaviside, INSERT_VALUES, user->Heaviside);
	
	DMDAVecGetArray(da, user->Grad_abs, &grad);
	DMDAVecGetArray(da, user->Heaviside, &h);
	
		
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				grad[k][j][to] = grad[k][j][from];
				h[k][j][to] = h[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				grad[k][j][to] = grad[k][j][from];
				h[k][j][to] = h[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				grad[k][to][i] = grad[k][from][i];
				h[k][to][i] = h[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				grad[k][to][i] = grad[k][from][i];
				h[k][to][i] = h[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				grad[to][j][i] = grad[from][j][i];
				h[to][j][i] = h[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				grad[to][j][i] = grad[from][j][i];
				h[to][j][i] = h[from][j][i];
			}
		}
	}
	
	
	// i direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(i==mx-1 || j==my-1 || k==mz-1) continue;
		if(j==0 || k==0) continue;
		
		double ajc = iaj[k][j][i];
		double csi0 = icsi[k][j][i].x, csi1 = icsi[k][j][i].y, csi2 = icsi[k][j][i].z;
		double eta0 = ieta[k][j][i].x, eta1 = ieta[k][j][i].y, eta2 = ieta[k][j][i].z;
		double zet0 = izet[k][j][i].x, zet1 = izet[k][j][i].y, zet2 = izet[k][j][i].z;
		
		double dldc, dlde, dldz;
		Compute_dscalar_i (i, j, k, mx, my, mz, level, nvert, &dldc, &dlde, &dldz );
		
		double dl_dx, dl_dy, dl_dz;
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
		double abs_grad = sqrt ( dl_dx*dl_dx + dl_dy*dl_dy + dl_dz*dl_dz );// + 1.e-20;
		
		//double abs_grad = 0.5 * ( grad[k][j][i] + grad[k][j][i+1] );
		//if(tistart==0 && ti<25) abs_grad+=1.e-20;

		double g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
		double g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
		double g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;
		
		if(abs_grad<1.e-10) curv[k][j][i].x=0.;
		else curv[k][j][i].x = (g11 * dldc + g21 * dlde + g31 * dldz) * ajc / abs_grad;
		if (!i_periodic && !ii_periodic && (i==0 || i==mx-2)) curv[k][j][i].x = 0;
		if ( nvert[k][j][i] + nvert[k][j][i+1] > 0.1 ) curv[k][j][i].x = 0;
	}
	
	// j direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(i==mx-1 || j==my-1 || k==mz-1) continue;
		if(i==0 || k==0) continue;

		double ajc = jaj[k][j][i];
		double csi0 = jcsi[k][j][i].x, csi1 = jcsi[k][j][i].y, csi2 = jcsi[k][j][i].z;
		double eta0= jeta[k][j][i].x, eta1 = jeta[k][j][i].y, eta2 = jeta[k][j][i].z;
		double zet0 = jzet[k][j][i].x, zet1 = jzet[k][j][i].y, zet2 = jzet[k][j][i].z;
		
		double dldc, dlde, dldz;
		Compute_dscalar_j (i, j, k, mx, my, mz, level, nvert, &dldc, &dlde, &dldz );
		
		
		double dl_dx, dl_dy, dl_dz;
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
		double abs_grad = sqrt ( dl_dx*dl_dx + dl_dy*dl_dy + dl_dz*dl_dz );// + 1.e-20;
		
		//double abs_grad = 0.5 * ( grad[k][j][i] + grad[k][j+1][i] );
		//if(tistart==0 && ti<25) abs_grad+=1.e-20;

		double g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
		double g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
		double g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;
		
		if(abs_grad<1.e-10) curv[k][j][i].y=0.;
		else curv[k][j][i].y = (g11 * dldc + g21 * dlde + g31 * dldz) * ajc / abs_grad;
		if(!j_periodic && !jj_periodic && (j==0 || j==my-2)) curv[k][j][i].y = 0;
		if ( nvert[k][j][i] + nvert[k][j+1][i] > 0.1 ) curv[k][j][i].y = 0;
	}
	
	// k direction
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(i==mx-1 || j==my-1 || k==mz-1) continue;
		if(i==0 || j==0) continue;
		
		double ajc = kaj[k][j][i];
		double csi0 = kcsi[k][j][i].x, csi1 = kcsi[k][j][i].y, csi2 = kcsi[k][j][i].z;
		double eta0 = keta[k][j][i].x, eta1 = keta[k][j][i].y, eta2 = keta[k][j][i].z;
		double zet0 = kzet[k][j][i].x, zet1 = kzet[k][j][i].y, zet2 = kzet[k][j][i].z;
		
		double dldc, dlde, dldz;
		Compute_dscalar_k (i, j, k, mx, my, mz, level, nvert, &dldc, &dlde, &dldz );
		
		double dl_dx, dl_dy, dl_dz;
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
		double abs_grad = sqrt ( dl_dx*dl_dx + dl_dy*dl_dy + dl_dz*dl_dz );// + 1.e-20;
		
		//double abs_grad = 0.5 * ( grad[k][j][i] + grad[k+1][j][i] );
		//if(tistart==0 && ti<25) abs_grad+=1.e-20;

		double g11 = csi0 * zet0 + csi1 * zet1 + csi2 * zet2;
		double g21 = eta0 * zet0 + eta1 * zet1 + eta2 * zet2;
		double g31 = zet0 * zet0 + zet1 * zet1 + zet2 * zet2;
		
		if(abs_grad<1.e-10) curv[k][j][i].z=0.;
		else curv[k][j][i].z = (g11 * dldc + g21 * dlde + g31 * dldz) * ajc / abs_grad;
		if(!k_periodic && !kk_periodic && (k==0 || k==mz-2)) curv[k][j][i].z = 0;
		if ( nvert[k][j][i] + nvert[k+1][j][i] > 0.1 ) curv[k][j][i].z = 0;
	}
		
	DMDAVecRestoreArray(fda, user->Curv, &curv);
		
	DMDALocalToLocalBegin(fda, user->Curv, INSERT_VALUES, user->Curv);
	DMDALocalToLocalEnd(fda, user->Curv, INSERT_VALUES, user->Curv);
		
	DMDAVecGetArray(fda, user->Curv, &curv);
	

	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				curv[k][j][to] = curv[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				curv[k][j][to] = curv[k][j][from];
			}
		}
	}
	
	if(ys==0 || ye==my) {
		int from, to;
				
		for (k=lzs; k<lze; k++)
		for (i=lxs; i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				curv[k][to][i] = curv[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				curv[k][to][i] = curv[k][from][i];
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int from, to;
		
		for (j=lys; j<lye; j++)
		for (i=lxs; i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				curv[to][j][i] = curv[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				curv[to][j][i] = curv[from][j][i];
			}
		}
	}
	
	
	DMDAVecRestoreArray(fda, user->Curv, &curv);
	DMDAVecRestoreArray(da, user->Grad_abs, &grad);
	DMDAVecRestoreArray(da, user->Heaviside, &h);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	
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
	
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	
}




// calculate the force of the sponge layer at the outlet 

void Compute_Force_SpongeLayer(UserCtx *user)
{
	DM 		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs, xe, ys, ye, zs, ze;
	int	mx, my, mz;
	int	i, j, k;
	int	lxs, lys, lzs, lxe, lye, lze;

	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  	mx = info.mx; my = info.my; mz = info.mz;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	Cmpnts	***csi, ***eta, ***zet, ***fsponge, ***ucat;
	PetscReal ***level, ***nvert, ***density, ***aj;
	
	
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lDensity, &density);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(da, user->lAj, &aj);

	DMDAVecGetArray(fda, user->lFSponge, &fsponge);
	DMDAVecGetArray(fda, user->lUcat, &ucat);
	
	/*
	// calculate the averaged velocity in the vertical (gravity) direction
	double vavg=0.0, count=0.0, vavg_sum=0.0, count_sum=0.0;	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		
		if (nvert[k][j][i] < 0.1 && k==mz-1-SpongeDistance) { 

			double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
			double Distrib = dH_1 ( level[k][j][i], dx );

			if (Distrib>1.e-6) {
				vavg+=ucat[k][j][i].y;
				count+=1.0;
			}
		}

	}
	
        MPI_Allreduce (&vavg, &vavg_sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce (&count, &count_sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

	
	double v_avg=vavg_sum/(count_sum+1.e-9);
	*/

	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		
		fsponge[k][j][i].x=0.0;
		fsponge[k][j][i].y=0.0;
		fsponge[k][j][i].z=0.0;

		if (nvert[k][j][i] < 0.1 && (k>=mz-1-SpongeDistance && k<mz-1)) { 

			double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
			double Distrib = DDelta ( level[k][j][i], dx );
			int kk = k-(mz-1-SpongeDistance);
			double Distrib1;
			if (kk<=0.5*SpongeDistance) {
				Distrib1 = 0.5*(1.0-cos((double)kk*M_PI/(0.5*(double)SpongeDistance)));
			}
			else Distrib1 = 1.0;
			

	      		double Fx = 0.0;
	      		double Fz = 0.0;
	      		double Fy = (0.0-ucat[k][j][i].y)/user->dt;
			Fy *= Distrib*Distrib1;

			fsponge[k][j][i].x=Fx*csi[k][j][i].x+Fy*csi[k][j][i].y+Fz*csi[k][j][i].z;
			fsponge[k][j][i].y=Fx*eta[k][j][i].x+Fy*eta[k][j][i].y+Fz*eta[k][j][i].z;
			fsponge[k][j][i].z=Fx*zet[k][j][i].x+Fy*zet[k][j][i].y+Fz*zet[k][j][i].z;

		}

		if (nvert[k][j][i] < 0.1 && (k>=0 && k<SpongeDistance)) { 

			double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
			double Distrib = DDelta ( level[k][j][i], dx );
			int kk = k;
			double Distrib1;
			if (kk>=0.5*SpongeDistance) {
				Distrib1 = 0.5*(1.0+cos((double)kk*M_PI/(0.5*(double)SpongeDistance)));
			}
			else Distrib1 = 1.0;
			

	      		double Fx = 0.0;
	      		double Fz = 0.0;
	      		double Fy = (0.0-ucat[k][j][i].y)/user->dt;
			Fy *= Distrib*Distrib1;

			fsponge[k][j][i].x=Fx*csi[k][j][i].x+Fy*csi[k][j][i].y+Fz*csi[k][j][i].z;
			fsponge[k][j][i].y=Fx*eta[k][j][i].x+Fy*eta[k][j][i].y+Fz*eta[k][j][i].z;
			fsponge[k][j][i].z=Fx*zet[k][j][i].x+Fy*zet[k][j][i].y+Fz*zet[k][j][i].z;

		}


	}
	
	
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lDensity, &density);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	
	DMDAVecRestoreArray(da, user->lAj, &aj);
	
	DMDAVecRestoreArray(fda, user->lFSponge, &fsponge);
	DMDAVecRestoreArray(fda, user->lUcat, &ucat);


  	DMLocalToGlobalBegin(fda, user->lFSponge, INSERT_VALUES, user->FSponge);
  	DMLocalToGlobalEnd(fda, user->lFSponge, INSERT_VALUES, user->FSponge);


	/*
	char fname[80];
	sprintf(fname,"FSponge");
	TECIOOut_rhs(user, user->FSponge, fname);
	*/
	

}


void Compute_Force_Subgridlevel(UserCtx *user, Vec Force_sublevel)
{
	DM 		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs, xe, ys, ye, zs, ze;
	int	mx, my, mz;
	int	i, j, k;
	int	lxs, lys, lzs, lxe, lye, lze;

	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;
  	mx = info.mx; my = info.my; mz = info.mz;
  
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	Cmpnts ***curv, ***csi, ***eta, ***zet;
	PetscReal ***aj;
	PetscReal ***level, ***nvert, ***density, ***grad, ***h, ***nu_t, ***force_sublevel;
	
	DMDAVecGetArray(fda, user->Curv, &curv);
	DMDAVecGetArray(da, user->Grad_abs, &grad);
	DMDAVecGetArray(da, user->Heaviside, &h);
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, user->lDensity, &density);
	DMDAVecGetArray(da, user->lNvert, &nvert);
		
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(da, user->lAj, &aj);
	
	if (les || rans) DMDAVecGetArray(da, user->lNu_t, &nu_t);
	DMDAVecGetArray(da, Force_sublevel, &force_sublevel);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double phi = level[k][j][i];
		double ajc = aj[k][j][i];
		double kappa = (curv[k][j][i].x - curv[k][j][i-1].x + curv[k][j][i].y - curv[k][j-1][i].y + curv[k][j][i].z - curv[k-1][j][i].z) * ajc;
		double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
		double gradH = dH (phi, dx); 
		
		double force = - 0.01*kappa*grad[k][j][i]*gradH*ajc;
		if ( nvert[k][j][i]> 0.1 ) force = 0.0;
		force_sublevel[k][j][i] += force;

	}
	
	DMDAVecRestoreArray(fda, user->Curv, &curv);
	DMDAVecRestoreArray(da, user->Grad_abs, &grad);
	DMDAVecRestoreArray(da, user->Heaviside, &h);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, user->lDensity, &density);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
			
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	
	if (les || rans) DMDAVecRestoreArray(da, user->lNu_t, &nu_t);

	DMDAVecRestoreArray(da, Force_sublevel, &force_sublevel);
	

	//char fname[80];
	//sprintf(fname,"Force_sublevel");
	//TECIOOut_rhs_da(user, Force_sublevel, fname);


}


void Levelset_smooth(UserCtx *user)
{
	DM		da = user->da;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze; // Local grid information
	int	mx, my, mz; // Dimensions in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***nvert, ***level, ***llevel, ***aj;
	Cmpnts ***cent;

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


	Vec Levelset_t;
	PetscReal ***level_t;
	VecDuplicate(user->lP, &Levelset_t);

	
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	DMDAVecGetArray(user->da, user->Levelset, &level);
	DMDAVecGetArray(user->da, user->lLevelset, &llevel);
	DMDAVecGetArray(user->fda, user->lCent, &cent);
	DMDAVecGetArray(user->da, user->lAj, &aj);
	DMDAVecGetArray(user->da, Levelset_t, &level_t);

	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {

		//if( nvert[k][j][i]>1.1) continue;
		double weight[3][3][3];
		double lev[3][3][3];

		//get_weight (i, j, k, mx, my, mz, aj, nvert, 1.1, weight);
		double usum, count;	
		usum=0.0; count=0.0;
		for(int p=-1; p<=1; p++)
		for(int q=-1; q<=1; q++)
		for(int r=-1; r<=1; r++) {
			int R=r+1, Q=q+1, P=p+1;
			int K=k+r, J=j+q, I=i+p;

			weight[R][Q][P] = 1./aj[K][J][I];
			lev[R][Q][P] = llevel[K][J][I];
			//usum += llevel[K][J][I];
			//count += 1;
		}
		

                //level_t[k][j][i] = usum/count;
                level_t[k][j][i] = integrate_testfilter_simpson(lev, weight);

	}

	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {

		//llevel[k][j][i] = level_t[k][j][i] ;
		llevel[k][j][i] = (1.0-smoothlevel)*llevel[k][j][i] + smoothlevel*level_t[k][j][i] ;

	}


	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
	DMDAVecRestoreArray(user->da, user->Levelset, &level);
	DMDAVecRestoreArray(user->da, user->lLevelset, &llevel);
	DMDAVecRestoreArray(user->fda, user->lCent, &cent);
	DMDAVecRestoreArray(user->da, user->lAj, &aj);
	DMDAVecRestoreArray(user->da, Levelset_t, &level_t);

	//DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
	//DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);

  	DMLocalToGlobalBegin(user->da, user->lLevelset, INSERT_VALUES, user->Levelset);
  	DMLocalToGlobalEnd(user->da, user->lLevelset, INSERT_VALUES, user->Levelset);



	VecDestroy(&Levelset_t);
}


