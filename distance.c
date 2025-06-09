#include "variables.h"

Vec LevelSet, LevelSet0, LevelSet_o;
double dtau;

//void Compute_dlevel_center_levelset	(int i,	int	j, int k,	 int mx, int my, int mz, double	sgn, int wall_distance,	PetscReal	***level,	PetscReal	***nvert,	double *dldc,	double *dlde,	double *dldz);

double d_levelset2()
{
  return d_parameter;
}

double eps_levelset2(double dx)
{
	return 0.5 * pow(dx, 1.-d_parameter);
}

double eps_levelset2()
{
  double d_parameter = d_levelset2();
  
	double eps = 0.5 * pow(dthick*dj_min*2, 1.-d_parameter);
	if(dthick_const>0) eps = 0.5 * pow(dthick_const, 1.-d_parameter);
	return eps;
}


void Init_LevelSet_Vectors(UserCtx *user)
{
	VecDuplicate(user->P, &LevelSet);
	VecDuplicate(user->P, &LevelSet0);
	VecDuplicate(user->P, &LevelSet_o);
	//VecDuplicate(user->lP, &lLevelSet);
};

void Destroy_LevelSet_Vectors(UserCtx	*user)
{
	VecDestroy(&LevelSet);
	VecDestroy(&LevelSet0);
	VecDestroy(&LevelSet_o);
	//VecDestroy(&lLevelSet);
};

// normal is calculated after the advection step and kept while the redistancing step
void Compute_LevelsetGrad0(UserCtx *user, Vec V)
{
	DM da = user->da, fda = user->fda;
	DMDALocalInfo info = user->info;
	int xs, xe, ys, ye, zs, ze;
	int mx, my, mz;
	int i, j, k;
	
	Vec Csi = user->lCsi, Eta = user->lEta, Zet = user->lZet;
	Vec Aj = user->lAj;

	Cmpnts	***csi, ***eta, ***zet, ***v;
	PetscReal	***aj, ***level0;
	PetscReal	***nvert;
		
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
	
	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;
	
	mx = info.mx; my = info.my; mz = info.mz;
	
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	VecSet(V, 0);
	
	Vec lLevelset0;
	VecDuplicate(user->lP, &lLevelset0);
	DMGlobalToLocalBegin(user->da, LevelSet0, INSERT_VALUES, lLevelset0);
	DMGlobalToLocalEnd(user->da, LevelSet0, INSERT_VALUES, lLevelset0);

	DMDAVecGetArray(da, lLevelset0, &level0);
	
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	//DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(fda, V, &v);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double dldc, dlde, dldz;
		double dl_dx, dl_dy, dl_dz;
		double ajc = aj[k][j][i];
		double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
		double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
		double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
		
		Compute_dscalar_center (i, j, k, mx, my, mz, level0, nvert, &dldc, &dlde, &dldz );
		Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
		double grad = sqrt ( dl_dx*dl_dx + dl_dy*dl_dy + dl_dz*dl_dz ) + 1.e-20;
		v[k][j][i].x = dl_dx / grad;
		v[k][j][i].y = dl_dy / grad;
		v[k][j][i].z = dl_dz / grad;
	}
	
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	//DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(fda, V, &v);
		
	DMDALocalToLocalBegin(fda, V, INSERT_VALUES, V);
	DMDALocalToLocalEnd(fda, V, INSERT_VALUES, V);
	
	DMDAVecGetArray(fda, V, &v);
	
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				v[k][j][to] = v[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				v[k][j][to] = v[k][j][from];
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
				
				v[k][to][i] = v[k][from][i];
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				v[k][to][i] = v[k][from][i];
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
				
				v[to][j][i] = v[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				v[to][j][i] = v[from][j][i];
			}
		}
	}
	DMDAVecRestoreArray(fda, V, &v);
	
	DMDAVecRestoreArray(da, lLevelset0, &level0);
	VecDestroy(&lLevelset0);
}

void Distance_Function_RHS (UserCtx *user, Vec Levelset_RHS, int wall_distance)
{
	DM da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;
	int	xs,	xe,	ys,	ye,	zs,	ze;
	int	mx,	my,	mz;
	int	i, j,	k;
	PetscReal	dpdc,	dpde,	dpdz;
	
	Vec Csi = user->lCsi, Eta = user->lEta, Zet = user->lZet;
	Vec Aj = user->lAj;

	Cmpnts	***csi, ***eta, ***zet, ***f1, ***f2, ***f3, ***grad;
	PetscReal	***aj, ***level, ***level0, ***rhs, ***grad_level;
	
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	PetscReal	***iaj, ***jaj, ***kaj;

	PetscReal	***nvert;
	
	Vec Grad, lLevelset0;	// grad	level, level0
	Vec F1, F2, F3;	// for levelset==2; F1 = phi(1-phi)nx, F2 = phi(1-phi)ny, F3 = phi(1-phi)nz, at the cell face (staggered)
	
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;
	
	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;
	
	mx = info.mx; my = info.my; mz = info.mz;
	
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	VecSet(Levelset_RHS, 0);
	
	VecDuplicate(user->lP, &Grad);
	if(levelset==2) {
		VecDuplicate(user->lUcont, &F1);
		VecDuplicate(user->lUcont, &F2);
		VecDuplicate(user->lUcont, &F3);
	}
	VecDuplicate(user->lP, &lLevelset0);
	
	DMGlobalToLocalBegin(user->da, LevelSet0, INSERT_VALUES, lLevelset0);
	DMGlobalToLocalEnd(user->da, LevelSet0, INSERT_VALUES, lLevelset0);
	
	DMDAVecGetArray(fda, Csi, &csi);
	DMDAVecGetArray(fda, Eta, &eta);
	DMDAVecGetArray(fda, Zet, &zet);
	DMDAVecGetArray(da,  Aj, &aj);
	
	DMDAVecGetArray(fda, user->lICsi, &icsi);
	DMDAVecGetArray(fda, user->lIEta, &ieta);
	DMDAVecGetArray(fda, user->lIZet, &izet);
	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(fda, user->lJEta, &jeta);
	DMDAVecGetArray(fda, user->lJZet, &jzet);
	DMDAVecGetArray(fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(fda, user->lKEta, &keta);
	DMDAVecGetArray(fda, user->lKZet, &kzet);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);
		
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lLevelset, &level);
	DMDAVecGetArray(da, lLevelset0, &level0);
	DMDAVecGetArray(da, Levelset_RHS, &rhs);
	
	double eps = eps_levelset2();
	
	if(levelset==1 || wall_distance) {
		DMDAVecGetArray(da,  Grad,  &grad_level);
		for (k=lzs;	k<lze; k++)
		for (j=lys;	j<lye; j++)
		for (i=lxs;	i<lxe; i++) {
			if(nvert[k][j][i]>0.1) {
				grad_level[k][j][i]=0;
			}
			else {
				double dldc, dlde, dldz;
				double dl_dx, dl_dy, dl_dz;
				
				double csi0=csi[k][j][i].x,csi1=csi[k][j][i].y, csi2=csi[k][j][i].z;
				double eta0=eta[k][j][i].x,eta1=eta[k][j][i].y, eta2=eta[k][j][i].z;
				double zet0=zet[k][j][i].x,zet1=zet[k][j][i].y, zet2=zet[k][j][i].z;
				double ajc = aj[k][j][i];
				
				double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
				double sgn = sign1(level0[k][j][i], dx);
				//if(wall_distance) sgn = sign(level0[k][j][i]);
				
				
				Compute_dlevel_center_levelset (i, j, k, mx, my, mz, sign(level0[k][j][i]), wall_distance, level, nvert, &dldc, &dlde, &dldz); //100521
				Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
				
				 // central diff
				/*
				Compute_dscalar_center (i, j, k, mx, my, mz, level0, nvert, &dldc, &dlde, &dldz );
				Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0,  zet1,  zet2,  ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
				*/
			
				grad_level[k][j][i] = sqrt( dl_dx*dl_dx + dl_dy*dl_dy + dl_dz*dl_dz );	// gradient with upwind method
			}
		}
		DMDAVecRestoreArray(da,  Grad,  &grad_level);
		DMDALocalToLocalBegin(user->da, Grad, INSERT_VALUES, Grad);
		DMDALocalToLocalEnd(user->da, Grad, INSERT_VALUES, Grad);
	}
	else if(levelset==2) {
		VecSet(F1, 0);
		VecSet(F2, 0);
		VecSet(F3, 0);
		
		DMDAVecGetArray(fda,  F1,  &f1);
		DMDAVecGetArray(fda,  F2,  &f2);
		DMDAVecGetArray(fda,  F3,  &f3);
		DMDAVecGetArray(fda,  user->LevelsetGrad,  &grad);
		
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
			
			double dldc, dlde, dldz;
					
			dldc = level[k][j][i+1] - level[k][j][i];

			if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
				dlde = (level[k][j  ][i+1] + level[k][j  ][i] - level[k][j-1][i+1] - level[k][j-1][i]) * 0.5;
			}
			else if  ((nvert[k][j-1][i])> 0.1 || (nvert[k][j-1][i+1])> 0.1) {
				dlde = (level[k][j+1][i+1] + level[k][j+1][i] - level[k][j  ][i+1] - level[k][j  ][i]) * 0.5;
			}
			else {
				dlde = (level[k][j+1][i+1] + level[k][j+1][i] - level[k][j-1][i+1] - level[k][j-1][i]) * 0.25;
			}	  

			if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
				dldz = (level[k  ][j][i+1] + level[k  ][j][i] - level[k-1][j][i+1] - level[k-1][j][i]) * 0.5;
			}
			else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j][i+1])> 0.1) {
				dldz = (level[k+1][j][i+1] + level[k+1][j][i] - level[k  ][j][i+1] - level[k  ][j][i]) * 0.5;
			}
			else {
				dldz = (level[k+1][j][i+1] + level[k+1][j][i] - level[k-1][j][i+1] - level[k-1][j][i]) * 0.25;
			}
			
			double dl_dx, dl_dy, dl_dz;
			Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
			
			double phi = 0.5 * (level[k][j][i] + level[k][j][i+1]);	// central differencing
			double nx = 0.5 * (grad[k][j][i].x + grad[k][j][i+1].x);
			double ny = 0.5 * (grad[k][j][i].y + grad[k][j][i+1].y);
			double nz = 0.5 * (grad[k][j][i].z + grad[k][j][i+1].z);
			
			f1[k][j][i].x = - phi * ( 1.0 - phi ) * nx + eps * dl_dx;
			f2[k][j][i].x = - phi * ( 1.0 - phi ) * ny + eps * dl_dy;
			f3[k][j][i].x = - phi * ( 1.0 - phi ) * nz + eps * dl_dz;
			
			/*
			double FL = level[k][j][i] * ( 1 - level[k][j][i]);
			double FR = level[k][j][i+1] * ( 1 - level[k][j][i+1]);

			f1[k][j][i].x = - 0.5 * ( FL * grad[k][j][i].x + FR * grad[k][j][i+1].x ) + eps * dl_dx;
			f2[k][j][i].x = - 0.5 * ( FL * grad[k][j][i].y + FR * grad[k][j][i+1].y ) + eps * dl_dy;
			f3[k][j][i].x = - 0.5 * ( FL * grad[k][j][i].z + FR * grad[k][j][i+1].z )+ eps * dl_dz;
			*/
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
			
			double dldc, dlde, dldz;
			
			if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
				dldc = (level[k][j+1][i  ] + level[k][j][i  ] - level[k][j+1][i-1] - level[k][j][i-1]) * 0.5;
			}
			else if ((nvert[k][j][i-1])> 0.1 || (nvert[k][j+1][i-1])> 0.1) {
				dldc = (level[k][j+1][i+1] + level[k][j][i+1] - level[k][j+1][i  ] - level[k][j][i  ]) * 0.5;
			}
			else {
				dldc = (level[k][j+1][i+1] + level[k][j][i+1] - level[k][j+1][i-1] - level[k][j][i-1]) * 0.25;
			}

			dlde = level[k][j+1][i] - level[k][j][i];
			
			if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
				dldz = (level[k  ][j+1][i] + level[k  ][j][i] - level[k-1][j+1][i] - level[k-1][j][i]) * 0.5;
			}
			else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j+1][i])> 0.1) {
				dldz = (level[k+1][j+1][i] + level[k+1][j][i] - level[k  ][j+1][i] - level[k  ][j][i]) * 0.5;
			}
			else {
				dldz = (level[k+1][j+1][i] + level[k+1][j][i] - level[k-1][j+1][i] - level[k-1][j][i]) * 0.25;
			}
			
			double dl_dx, dl_dy, dl_dz;
			Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);

			
			double phi = 0.5 * (level[k][j][i] + level[k][j+1][i]);	// central differencing
			double nx = 0.5 * (grad[k][j][i].x + grad[k][j+1][i].x);
			double ny = 0.5 * (grad[k][j][i].y + grad[k][j+1][i].y);
			double nz = 0.5 * (grad[k][j][i].z + grad[k][j+1][i].z);
			
			f1[k][j][i].y = - phi * ( 1.0 - phi ) * nx + eps * dl_dx;
			f2[k][j][i].y = - phi * ( 1.0 - phi ) * ny + eps * dl_dy;
			f3[k][j][i].y = - phi * ( 1.0 - phi ) * nz + eps * dl_dz;
			/*
			double FL = level[k][j][i] * ( 1 - level[k][j][i]);
			double FR = level[k][j+1][i] * ( 1 - level[k][j+1][i]);

			f1[k][j][i].y = - 0.5 * ( FL * grad[k][j][i].x + FR * grad[k][j+1][i].x ) + eps * dl_dx;
			f2[k][j][i].y = - 0.5 * ( FL * grad[k][j][i].y + FR * grad[k][j+1][i].y ) + eps * dl_dy;
			f3[k][j][i].y = - 0.5 * ( FL * grad[k][j][i].z + FR * grad[k][j+1][i].z ) + eps * dl_dz;
			*/
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
			
			double dldc, dlde, dldz;
			
			if ((nvert[k][j][i+1])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
				dldc = (level[k+1][j][i  ] + level[k][j][i  ] - level[k+1][j][i-1] - level[k][j][i-1]) * 0.5;
			}
			else if ((nvert[k][j][i-1])> 0.1 || (nvert[k+1][j][i-1])> 0.1) {
				dldc = (level[k+1][j][i+1] + level[k][j][i+1] - level[k+1][j][i  ] - level[k][j][i  ]) * 0.5;
			}
			else {
				dldc = (level[k+1][j][i+1] + level[k][j][i+1] - level[k+1][j][i-1] - level[k][j][i-1]) * 0.25;
			}

			if ((nvert[k][j+1][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
				dlde = (level[k+1][j  ][i] + level[k][j  ][i] - level[k+1][j-1][i] - level[k][j-1][i]) * 0.5;
			}
			else if ((nvert[k][j-1][i])> 0.1 || (nvert[k+1][j-1][i])> 0.1) {
				dlde = (level[k+1][j+1][i] + level[k][j+1][i] - level[k+1][j  ][i] - level[k][j  ][i]) * 0.5;
			}
			else {
				dlde = (level[k+1][j+1][i] + level[k][j+1][i] - level[k+1][j-1][i] - level[k][j-1][i]) * 0.25;
			}

			dldz = level[k+1][j][i] - level[k][j][i];
			
			
			double dl_dx, dl_dy, dl_dz;
			Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dldc, dlde, dldz, &dl_dx, &dl_dy, &dl_dz);
			
			double phi = 0.5 * (level[k][j][i] + level[k+1][j][i]);	// central differencing
			double nx = 0.5 * (grad[k][j][i].x + grad[k+1][j][i].x);
			double ny = 0.5 * (grad[k][j][i].y + grad[k+1][j][i].y);
			double nz = 0.5 * (grad[k][j][i].z + grad[k+1][j][i].z);
			
			f1[k][j][i].z = - phi * ( 1.0 - phi ) * nx + eps * dl_dx;
			f2[k][j][i].z = - phi * ( 1.0 - phi ) * ny + eps * dl_dy;
			f3[k][j][i].z = - phi * ( 1.0 - phi ) * nz + eps * dl_dz;
			/*
			double FL = level[k][j][i] * ( 1 - level[k][j][i]);
			double FR = level[k+1][j][i] * ( 1 - level[k+1][j][i]);

			f1[k][j][i].z = - 0.5 * ( FL * grad[k][j][i].x + FR * grad[k+1][j][i].x ) + eps * dl_dx;
			f2[k][j][i].z = - 0.5 * ( FL * grad[k][j][i].y + FR * grad[k+1][j][i].y ) + eps * dl_dy;
			f3[k][j][i].z = - 0.5 * ( FL * grad[k][j][i].z + FR * grad[k+1][j][i].z ) + eps * dl_dz;
			*/
		}

		DMDAVecRestoreArray(fda,  F1,  &f1);
		DMDAVecRestoreArray(fda,  F2,  &f2);
		DMDAVecRestoreArray(fda,  F3,  &f3);
		DMDAVecRestoreArray(fda,  user->LevelsetGrad,  &grad);
	
		DMDALocalToLocalBegin(user->fda, F1, INSERT_VALUES, F1);
		DMDALocalToLocalEnd(user->fda, F1, INSERT_VALUES, F1);
		DMDALocalToLocalBegin(user->fda, F2, INSERT_VALUES, F2);
		DMDALocalToLocalEnd(user->fda, F2, INSERT_VALUES, F2);
		DMDALocalToLocalBegin(user->fda, F3, INSERT_VALUES, F3);
		DMDALocalToLocalEnd(user->fda, F3, INSERT_VALUES, F3);
	}
	
	DMDAVecGetArray(da,  Grad,  &grad_level);
	
	if(levelset==2) {
		DMDAVecGetArray(fda,  F1,  &f1);
		DMDAVecGetArray(fda,  F2,  &f2);
		DMDAVecGetArray(fda,  F3,  &f3);
	}
	
	// Neumann, periodic conditions
	if(xs==0 || xe==mx) {
		int from, to;
		for (k=lzs; k<lze; k++)
		for (j=lys; j<lye; j++) {
			if(xs==0) {
				i=1, from=i, to	= 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from = -2;
				
				grad_level[k][j][to] = grad_level[k][j][from];
				if(levelset==2) {
					f1[k][j][to] = f1[k][j][from];
					f2[k][j][to] = f2[k][j][from];
					f3[k][j][to] = f3[k][j][from];
				}
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from = mx+1;
				
				grad_level[k][j][to] = grad_level[k][j][from];
				if(levelset==2) {
					f1[k][j][to] = f1[k][j][from];
					f2[k][j][to] = f2[k][j][from];
					f3[k][j][to] = f3[k][j][from];
				}
			}
		}
	}
	
	if(ys==0|| ye==my) {
		int	from, to;
				
		for (k=lzs;	k<lze; k++)
		for (i=lxs;	i<lxe; i++) {
			if(ys==0) {
				j = 1, from = j, to = 0;
				
				if(j_periodic) from = my-2;
				else if(jj_periodic) from = -2;
				
				grad_level[k][to][i] = grad_level[k][from][i];
				if(levelset==2) {
					f1[k][to][i] = f1[k][from][i];
					f2[k][to][i] = f2[k][from][i];
					f3[k][to][i] = f3[k][from][i];
				}
			}
			
			if(ye==my) {
				j = my-2, from = j, to = my-1;
				
				if(j_periodic) from = 1;
				else if(jj_periodic) from = my+1;
				
				grad_level[k][to][i] = grad_level[k][from][i];
				if(levelset==2) {
					f1[k][to][i] = f1[k][from][i];
					f2[k][to][i] = f2[k][from][i];
					f3[k][to][i] = f3[k][from][i];
				}
			}
		}
	}
	
	if(zs==0 || ze==mz) {
		int	from, to;
		
		for (j=lys;	j<lye; j++)
		for (i=lxs;	i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				grad_level[to][j][i] = grad_level[from][j][i];
				if(levelset==2) {
					f1[to][j][i] = f1[from][j][i];
					f2[to][j][i] = f2[from][j][i];
					f3[to][j][i] = f3[from][j][i];
				}
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				grad_level[to][j][i] = grad_level[from][j][i];
				if(levelset==2) {
					f1[to][j][i] = f1[from][j][i];
					f2[to][j][i] = f2[from][j][i];
					f3[to][j][i] = f3[from][j][i];
				}
			}
		}
	}
	
	DMDAVecRestoreArray(da,  Grad,  &grad_level);

	DMDALocalToLocalBegin(user->da, Grad, INSERT_VALUES, Grad);
	DMDALocalToLocalEnd(user->da, Grad, INSERT_VALUES, Grad);

	if(levelset==2) {
		DMDAVecRestoreArray(fda,  F1,  &f1);
		DMDAVecRestoreArray(fda,  F2,  &f2);
		DMDAVecRestoreArray(fda,  F3,  &f3);
		
		DMDALocalToLocalBegin(user->fda, F1, INSERT_VALUES, F1);
		DMDALocalToLocalEnd(user->fda, F1, INSERT_VALUES, F1);
		DMDALocalToLocalBegin(user->fda, F2, INSERT_VALUES, F2);
		DMDALocalToLocalEnd(user->fda, F2, INSERT_VALUES, F2);
		DMDALocalToLocalBegin(user->fda, F3, INSERT_VALUES, F3);
		DMDALocalToLocalEnd(user->fda, F3, INSERT_VALUES, F3);
		
		DMDAVecGetArray(fda,  F1,  &f1);
		DMDAVecGetArray(fda,  F2,  &f2);
		DMDAVecGetArray(fda,  F3,  &f3);
	}
	
	DMDAVecGetArray(da,  Grad,  &grad_level);
	

	for(k=zs; k<ze; k++)
	for(j=ys; j<ye; j++)
	for(i=xs; i<xe; i++) {
		if (i<= 0 || i>= mx-1 || j<=0 || j>=my-1 || k<=0 || k>=mz-1 || nvert[k][j][i]>0.1){
			rhs[k][j][i]=0.;
			continue;
		}
				
		if(wall_distance) {
			if(nvert[k][j][i]>0.1) { rhs[k][j][i]=0.; continue; }
			if(i <= 1 && (user->bctype[0]==1 || user->bctype[0]==-1 || user->bctype[0]==-2)) { rhs[k][j][i]=0.; continue; }
			if(i >=mx-2 &&	(user->bctype[1]==1 || user->bctype[1]==-1 || user->bctype[1]==-2)) { rhs[k][j][i]=0.; continue; }
			if(j <=1 && (user->bctype[2]==1 || user->bctype[2]==-1 || user->bctype[2]==-2)) { rhs[k][j][i]=0.; continue; }
			if(j >=my-2 && (user->bctype[3]==1 || user->bctype[3]==-1 || user->bctype[3]==-2 || user->bctype[3]==12)) { rhs[k][j][i]=0.; continue; }
			if(k<=1 && (user->bctype[4]==1 || user->bctype[4]==-1 || user->bctype[4]==-2)) { rhs[k][j][i]=0.; continue; }
			if(k>=mz-2 && (user->bctype[5]==1 || user->bctype[5]==-1 || user->bctype[5]==-2)){ rhs[k][j][i]=0.; continue; }
		}
		
		double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
		double eta0 = eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
		double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
		double ajc = aj[k][j][i];
		
		if(levelset==1 || wall_distance) {
			double denom[3][3][3], num[3][3][3], weight[3][3][3];
			
			for(int p=-1; p<=1; p++)
			for(int q=-1; q<=1; q++)
			for(int r=-1; r<=1; r++) {
				int R=r+1, Q=q+1, P=p+1;
				int K=k+r, J=j+q, I=i+p;
				double phi = level[K][J][I], grad=grad_level[K][J][I];//, dx=pow(1./aj[K][J][I],1./3.);
				double dx = levelset_thickness ( aj[K][J][I], csi[K][J][I], eta[K][J][I], zet[K][J][I] );
				double f = dH(phi,dx)*grad;
				
				double _sgn = sign1( level0[K][J][I], dx );
			
				num[R][Q][P] = dH(phi,dx) * _sgn * ( 1.	- grad );
				denom[R][Q][P] = dH(phi,dx) * f;
			}
			
			for(int p=-1; p<=1; p++)
			for(int q=-1; q<=1; q++)
			for(int r=-1; r<=1; r++) {
				int R=r+1, Q=q+1, P=p+1;
				int K=k+r, J=j+q, I=i+p;
				if( nvert[K][J][I]>0.1) {
					num[R][Q][P] = num[1][1][1];
					denom[R][Q][P] = denom[1][1][1];
				}
			}
			
			get_weight (i, j, k, mx, my, mz, aj, nvert, 0.1, weight);
			
			double numerator = integrate_testfilter_simpson(num, weight);
			double denominator = integrate_testfilter_simpson(denom, weight);
			
			double correction;
			
			if( fabs(denominator)<1.e-10 ) correction=0;
			else {
				double grad = grad_level[k][j][i];
				double phi = level[k][j][i];
				double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
				double f = dH(phi,dx)*grad;
				correction = - numerator / denominator;
				correction *= dH(phi,dx) * grad;
			}
			
			double dlevel_dx, dlevel_dy, dlevel_dz;
			double dldc, dlde, dldz;
			
			Compute_dlevel_center_levelset (i, j, k, mx, my, mz, sign(level0[k][j][i]), wall_distance, level, nvert, &dldc, &dlde, &dldz); //100521
			Compute_dscalar_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dldc, dlde, dldz, &dlevel_dx, &dlevel_dy, &dlevel_dz);
			
			double dx = levelset_thickness ( aj[k][j][i], csi[k][j][i], eta[k][j][i], zet[k][j][i] );
			double sgn = sign1(level0[k][j][i],dx);
			rhs[k][j][i] = sgn * ( 1. - sqrt( dlevel_dx*dlevel_dx + dlevel_dy*dlevel_dy + dlevel_dz*dlevel_dz ) );
			
			if(!wall_distance) rhs[k][j][i] += correction;	// Sussman Fetami
		}
		else if (levelset==2) {
			rhs[k][j][i] = csi0 * (f1[k][j][i].x-f1[k][j][i-1].x);
			rhs[k][j][i] += eta0 * (f1[k][j][i].y-f1[k][j-1][i].y);
			rhs[k][j][i] += zet0 * (f1[k][j][i].z-f1[k-1][j][i].z);
			
			rhs[k][j][i] += csi1 * (f2[k][j][i].x-f2[k][j][i-1].x);
			rhs[k][j][i] += eta1 * (f2[k][j][i].y-f2[k][j-1][i].y);
			rhs[k][j][i] += zet1 * (f2[k][j][i].z-f2[k-1][j][i].z);
			
			rhs[k][j][i] += csi2 * (f3[k][j][i].x-f3[k][j][i-1].x);
			rhs[k][j][i] += eta2 * (f3[k][j][i].y-f3[k][j-1][i].y);
			rhs[k][j][i] += zet2 * (f3[k][j][i].z-f3[k-1][j][i].z);
			
			rhs[k][j][i] *= ajc;
			
			//printf("%f %f %f\n", flux[k][j][i].x, flux[k][j][i].y, flux[k][j][i].z);
		}
	}

	DMDAVecRestoreArray(da,  Grad,  &grad_level);
	if(levelset==2) {
		DMDAVecRestoreArray(fda,  F1,  &f1);
		DMDAVecRestoreArray(fda,  F2,  &f2);
		DMDAVecRestoreArray(fda,  F3,  &f3);
	}
	DMDAVecRestoreArray(fda, Csi, &csi);
	DMDAVecRestoreArray(fda, Eta, &eta);
	DMDAVecRestoreArray(fda, Zet, &zet);
	DMDAVecRestoreArray(da,  Aj, &aj);
	
	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(fda, user->lIZet, &izet);
	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(fda, user->lJZet, &jzet);
	DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(fda, user->lKEta, &keta);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lLevelset, &level);
	DMDAVecRestoreArray(da, lLevelset0, &level0);
	DMDAVecRestoreArray(da, Levelset_RHS, &rhs);
	
	VecDestroy(&Grad);
	if(levelset==2) {
		VecDestroy(&F1);
		VecDestroy(&F2);
		VecDestroy(&F3);
	}
	VecDestroy(&lLevelset0);
}

void Distance_Function_IC(UserCtx	*user)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze;	// Local grid	information
	int	mx, my, mz;	// Dimensions	in three directions
	int	i, j, k, ibi;
	int	lxs, lxe, lys, lye, lzs, lze;
	Cmpnts	***csi, ***eta, ***zet;
	PetscReal	***nvert, ***L0, ***aj;

	DMDAGetLocalInfo(da, &info);
	mx = info.mx;	my = info.my;	mz = info.mz;
	xs = info.xs;	xe = xs	+	info.xm;
	ys = info.ys;	ye = ys	+	info.ym;
	zs = info.zs;	ze = zs	+	info.zm;

	lxs = xs;	lxe = xe;
	lys = ys;	lye = ye;
	lzs = zs;	lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, LevelSet0, &L0);
	
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(da, user->lAj, &aj);
	
	
	if(immersed)
	for(ibi=0; ibi<NumberOfBodies; ibi++)
	{
		//IBMNodes *ibm = &ibm_ptr[ibi];
		IBMListNode	*current;
		current = user->ibmlist[ibi].head;
		while	(current) {
			IBMInfo	*ibminfo = &current->ibm_intp;
			current	= current->next;
			double sb = ibminfo->d_s;
			i=ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
			L0[k][j][i] = sb;
		};
	}	
	
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int	flag=0;
		double dist=1.e10;
		
		if(nvert[k][j][i]>0.1) continue;
		
		if((user->bctype[0]==1|| user->bctype[0]==-1 || user->bctype[0]==-2) && i==1){
			double area=sqrt(csi[k][j][i].x*csi[k][j][i].x+csi[k][j][i].y*csi[k][j][i].y+csi[k][j][i].z*csi[k][j][i].z);
			dist = PetscMin(dist,0.5/aj[k][j][i]/area);
			flag=1;
			L0[k][j][i-1] = dist;
		}
		
		
		if((user->bctype[1]==1|| user->bctype[1]==-1 || user->bctype[1]==-2) && i==mx-2 ) {
			double area=sqrt(csi[k][j][i].x*csi[k][j][i].x+csi[k][j][i].y*csi[k][j][i].y+csi[k][j][i].z*csi[k][j][i].z);
			dist = PetscMin(dist,0.5/aj[k][j][i]/area);
			flag=1;
			L0[k][j][i+1] = dist;
		}
		if((user->bctype[2]==1|| user->bctype[2]==-1 || user->bctype[2]==-2) && j==1){
			double area=sqrt(eta[k][j][i].x*eta[k][j][i].x+eta[k][j][i].y*eta[k][j][i].y+eta[k][j][i].z*eta[k][j][i].z);
			dist = PetscMin(dist,0.5/aj[k][j][i]/area);
			flag=1;
			L0[k][j-1][i] = dist;
		}
		if((user->bctype[3]==1|| user->bctype[3]==-1 || user->bctype[3]==-2 || user->bctype[3]==12) &&j==my-2){
			double area=sqrt(eta[k][j][i].x*eta[k][j][i].x+eta[k][j][i].y*eta[k][j][i].y+eta[k][j][i].z*eta[k][j][i].z);
			dist = PetscMin(dist,0.5/aj[k][j][i]/area);
			flag=1;
			L0[k][j+1][i] = dist;
		}
		if((user->bctype[4]==1|| user->bctype[4]==-1 || user->bctype[4]==-2) && k==1){
			double area=sqrt(zet[k][j][i].x*zet[k][j][i].x+zet[k][j][i].y*zet[k][j][i].y+zet[k][j][i].z*zet[k][j][i].z);
			dist = PetscMin(dist,0.5/aj[k][j][i]/area);
			flag=1;
			L0[k-1][j][i] = dist;
		}
		if((user->bctype[5]==1 || user->bctype[5]==-1 || user->bctype[5]==-2) && k==mz-2 ) {
			double area=sqrt(zet[k][j][i].x*zet[k][j][i].x+zet[k][j][i].y*zet[k][j][i].y+zet[k][j][i].z*zet[k][j][i].z);
			dist = PetscMin(dist,0.5/aj[k][j][i]/area);
			flag=1;
			L0[k+1][j][i] = dist;
		}

		if(flag) L0[k][j][i] = dist;
		else L0[k][j][i] = 0.05;
		
		//if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ) L0[k][j][i]=0;
	}
	

	
	
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, LevelSet0, &L0);
	
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(da, user->lAj, &aj);
	
	VecCopy	(LevelSet0, LevelSet); 
	VecCopy	(LevelSet0, LevelSet_o); 
};

PetscErrorCode FormFunction_Distance(SNES snes, Vec L, Vec Rhs, void *ptr)
{
	UserCtx	*user = (UserCtx*)ptr;

	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze;	// Local grid	information
	int	mx, my, mz;	// Dimensions	in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***level;

	DMDAGetLocalInfo(user->da, &info);
	mx = info.mx;	my = info.my;	mz = info.mz;
	xs = info.xs;	xe = xs	+	info.xm;
	ys = info.ys;	ye = ys	+	info.ym;
	zs = info.zs;	ze = zs	+	info.zm;

	lxs = xs;	lxe = xe;
	lys = ys;	lye = ye;
	lzs = zs;	lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	DMGlobalToLocalBegin(user->da, L, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, L, INSERT_VALUES, user->lLevelset);
	
	DMDAVecGetArray(user->da, user->lLevelset, &level);
	
	if(xs==0 || xe==mx) {
		int	from, to;
		for (k=lzs;	k<lze; k++)
		for (j=lys;	j<lye; j++) {
			if(xs==0) {
				i = 1, from = i, to = 0;
				
				if(i_periodic) from = mx-2;
				else if(ii_periodic) from	= -2;
				
				level[k][j][to] = level[k][j][from];
			}
			
			if(xe==mx) {
				i = mx-2, from = i, to = mx-1;
				
				if(i_periodic) from = 1;
				else if(ii_periodic) from	= mx+1;
				
				level[k][j][to] = level[k][j][from];
			}
		}
	}
	
	if(ys==0 ||	ye==my) {
		int	from, to;
				
		for (k=lzs;	k<lze; k++)
		for (i=lxs;	i<lxe; i++) {
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
	
	if(zs==0 ||	ze==mz) {
		int	from, to;
		
		for (j=lys;	j<lye; j++)
		for (i=lxs;	i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				level[to][j][i] = level[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				level[to][j][i] = level[from][j][i];
			}
		}
	}
	
	DMDAVecRestoreArray(user->da, user->lLevelset, &level);
	
	Distance_Function_RHS(user, Rhs, 1);
	VecAXPY(Rhs, -1/dtau, L);
	VecAXPY(Rhs, 1/dtau, LevelSet_o);
	
	return(0);
}

void Solve_Distance_Explicit(UserCtx *user)
{
	DMDALocalInfo	info;
	int	xs, xe, ys, ye, zs, ze;	// Local grid	information
	int	mx, my, mz;	// Dimensions	in three directions
	int	i, j, k;
	int	lxs, lxe, lys, lye, lzs, lze;
	PetscReal	***level;

	DMDAGetLocalInfo(user->da, &info);
	mx = info.mx;	my = info.my;	mz = info.mz;
	xs = info.xs;	xe = xs	+	info.xm;
	ys = info.ys;	ye = ys	+	info.ym;
	zs = info.zs;	ze = zs	+	info.zm;

	lxs = xs;	lxe = xe;
	lys = ys;	lye = ye;
	lzs = zs;	lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	DMGlobalToLocalBegin(user->da, LevelSet, INSERT_VALUES, user->lLevelset);
	DMGlobalToLocalEnd(user->da, LevelSet, INSERT_VALUES, user->lLevelset);
	
	DMDAVecGetArray(user->da, user->lLevelset, &level);
	
	if(xs==0 ||	xe==mx) {
		int	from, to;
		for (k=lzs;	k<lze; k++)
		for (j=lys;	j<lye; j++) {
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
	
	if(ys==0 ||	ye==my) {
		int	from, to;
				
		for (k=lzs;	k<lze; k++)
		for (i=lxs;	i<lxe; i++) {
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
	
	if(zs==0 ||	ze==mz) {
		int	from, to;
		
		for (j=lys;	j<lye; j++)
		for (i=lxs;	i<lxe; i++) {
			if(zs==0) {
				k = 1, from = k, to = 0;
				
				if(k_periodic) from = mz-2;
				else if(kk_periodic) from = -2;
				
				level[to][j][i]	= level[from][j][i];
			}
			
			if(ze==mz) {
				k = mz-2, from = k, to = mz-1;
				
				if(k_periodic) from = 1;
				else if(kk_periodic) from = mz+1;
				
				level[to][j][i] = level[from][j][i];
			}
		}
	}
	
	DMDAVecRestoreArray(user->da, user->lLevelset, &level);
	
	Vec Rhs;
	VecDuplicate(user->P, &Rhs);
	Distance_Function_RHS(user, Rhs, 1);
	VecAXPY(LevelSet, dtau, Rhs);
	VecDestroy(&Rhs);
	return;
}

void Solve_Distance(UserCtx	*user, int iter)
{
	SNES snes_distance;
	KSP ksp;
	PC pc;
	Vec	r;
	Mat	J;
	double norm;
	
	int	bi=0;
		
	VecDuplicate(LevelSet, &r);
	
	
	SNESCreate(PETSC_COMM_WORLD,&snes_distance);
	SNESSetFunction(snes_distance,r,FormFunction_Distance,(void	*)&user[bi]);
	MatCreateSNESMF(snes_distance, &J);
	SNESSetJacobian(snes_distance,J,J,MatMFFDComputeJacobian,(void *)&user[bi]);
		
	
	SNESSetType(snes_distance, SNESTR);			//SNESTR,SNESLS	
	double tol=1.e-2;
	SNESSetMaxLinearSolveFailures(snes_distance,10000);
	SNESSetMaxNonlinearStepFailures(snes_distance,10000);		
	SNESKSPSetUseEW(snes_distance, PETSC_TRUE);
	SNESKSPSetParametersEW(snes_distance,3,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	SNESSetTolerances(snes_distance,PETSC_DEFAULT,tol,PETSC_DEFAULT,5,50000);	// snes	iter
		
	SNESGetKSP(snes_distance, &ksp);
	KSPSetType(ksp, KSPGMRES);
	//KSPGMRESSetPreAllocateVectors(ksp);
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCNONE);
	
	//int	maxits=10;	
	int	maxits=4;	// ksp iter
	double rtol=tol, atol=PETSC_DEFAULT, dtol=PETSC_DEFAULT;
	KSPSetTolerances(ksp,rtol,atol,dtol,maxits);
	
	SNESMonitorSet(snes_distance,MySNESMonitor,PETSC_NULL,PETSC_NULL);
	SNESSolve(snes_distance, PETSC_NULL, LevelSet);
	
	SNESGetFunctionNorm(snes_distance, &norm);
	//PetscPrintf(PETSC_COMM_WORLD, "\nDistance	SNES residual	norm=%.5e\n\n", norm);
	VecDestroy(&r);
	MatDestroy(&J);
	SNESDestroy(&snes_distance);
};

void Compute_Distance_Function(UserCtx *user)
{
	DMDALocalInfo	info;
	int	i, j, k;
	DMDAGetLocalInfo(user->da, &info);
	int	mx = info.mx, my = info.my, mz = info.mz;
	int	xs = info.xs, xe = xs	+	info.xm;
	int	ys = info.ys, ye = ys	+	info.ym;
	int	zs = info.zs, ze = zs	+	info.zm;
	int	lxs = xs, lxe = xe;
	int	lys = ys, lye = ye;
	int	lzs = zs, lze = ze;
	PetscReal	***level;
	
	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	int	iter=0;
	Init_LevelSet_Vectors(user);
	Distance_Function_IC(user);
		
	Vec	D;
	double norm=100, norm_old;
		
	dtau = dx_min*0.5;
	//dtau = dx_min	*	0.5;
	
	VecDuplicate(LevelSet, &D);
	
	do {
		norm_old = norm;
		
		PetscPrintf(PETSC_COMM_WORLD, "\nSolving Distance %d... (dtau=%f)\n", iter, dtau);
		//Solve_Distance_Explicit(user);
		Solve_Distance(user, iter);
		VecWAXPY(D, -1., LevelSet_o, LevelSet);
		VecNorm(D, NORM_INFINITY, &norm);
		PetscPrintf(PETSC_COMM_WORLD, "\nNorm=%.5e\n\n", norm);
		
		PetscReal ***level;
		DMDAVecGetArray(user->da, LevelSet, &level);
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) if(level[k][j][i]<0) level[k][j][i]=0.01;
		DMDAVecRestoreArray(user->da, LevelSet, &level);
	
		if(iter>10 && (fabs(norm)<1.e-5 || iter>1000) ) break;
		if(iter>10 && iter%50==0) dtau *=1.025;
		VecCopy(LevelSet, LevelSet_o);
		
		iter++;
	}	while(1);
	VecDestroy(&D);
	
	Vec	lLevelset;
	PetscReal	***llevel;
	VecDuplicate(user->lP, &lLevelset);

	DMGlobalToLocalBegin(user->da, LevelSet, INSERT_VALUES, lLevelset);
	DMGlobalToLocalEnd(user->da, LevelSet, INSERT_VALUES, lLevelset);

	DMDAVecGetArray(user->da, LevelSet, &level);
	DMDAVecGetArray(user->da, lLevelset, &llevel);

	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int c=k, b=j, a=i, flag=0;
		
		// Neumann conditions
		if(i==0) a=1, flag=1;
		if(i==mx-1) a=mx-2, flag=1;
		if(j==0) b=1, flag=1;
		if(j==my-1) b=my-2, flag=1;
		if(k==0) c=1, flag=1;
		if(k==mz-1) c=mz-2, flag=1;
		
		if(i_periodic && i==0) a=mx-2, flag=1;
		else if(i_periodic &&	i==mx-1) a=1, flag=1;
		
		if(j_periodic && j==0) b=my-2, flag=1;
		else if(j_periodic &&	j==my-1) b=1, flag=1;
		
		if(k_periodic && k==0) c=mz-2, flag=1;
		else if(k_periodic &&	k==mz-1) c=1, flag=1;
		
		
		if(ii_periodic && i==0) a=-2, flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
		
		if(jj_periodic && j==0) b=-2, flag=1;
		else if(jj_periodic && j==my-1) b=my+1, flag=1;
		
		if(kk_periodic && k==0) c=-2, flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
		
		if(flag) level[k][j][i] = llevel[c][b][a];
	}
	DMDAVecRestoreArray(user->da, LevelSet, &level);
	DMDAVecRestoreArray(user->da, lLevelset, &llevel);

	VecDestroy(&lLevelset);
	VecCopy(LevelSet, user->Distance);	
	Destroy_LevelSet_Vectors(user);
};

void Compute_dlevel_center_levelset (int i, int j, int k,  int mx, int my, int mz, double sgn, int wall_distance, PetscReal ***level, PetscReal ***nvert, double *dldc, double *dlde, double *dldz)
{
	double dplus0, dminus0;
	double _dplus, _dminus;
	double f=0.0; //second-order
	//		f=0;
	
	if(wall_distance) f=0;

	// i-direction
	int iLL=i-2, iL=i-1, iR=i+1, iRR=i+2;
	
	if (nvert[k][j][i-1]>0.1 ) {
	  //iL = i;
		iLL = iL;
	}
	if ( i!=1 && nvert[k][j][i-2]>1.1 ) {
		iLL = iL;
	}
	if (nvert[k][j][i+1]>0.1 ) {
	  //iR = i;
		iRR = iR;
	}
	if ( i!=mx-2 && nvert[k][j][i+2]>1.1 ) {
		iRR = iR;
	}
	
	if(i==1) {
		if(i_periodic) iLL = mx-3;
		else if(ii_periodic) iLL=-3;
		else iLL = iL;
	}
	else if(i==mx-2) {
		if(i_periodic) iRR = 2;
		else if(ii_periodic) iRR=mx+2;
		else iRR = iR;
	}
	
	dplus0=level[k][j][iR]-level[k][j][i], dminus0=level[k][j][i]-level[k][j][iL];
	
	_dplus  =  dplus0 - 0.5 * M(level[k][j][iR]-2.*level[k][j][i]+level[k][j][iL], level[k][j][iRR]-2.*level[k][j][iR]+level[k][j][i]) * f;
	_dminus = dminus0 + 0.5 * M(level[k][j][iR]-2.*level[k][j][i]+level[k][j][iL], level[k][j][i]-2.*level[k][j][iL]+level[k][j][iLL]) * f;
		
	if( sgn*dplus0<0 && sgn*(dminus0+dplus0)<0) *dldc=_dplus;
	else if( sgn*dminus0>0 && sgn*(dplus0+dminus0)>0) *dldc=_dminus;
	else if( sgn*dminus0<0 && sgn*dplus0>0) *dldc=0;
	else *dldc=0.5*(_dplus+_dminus);
		
	// j-direction
	int jLL=j-2, jL=j-1, jR=j+1, jRR=j+2;
	
	if (nvert[k][j-1][i]>0.1 ) {
	  //jL = j;
		jLL = jL;
	}
	if ( j!=1 && nvert[k][j-2][i]>1.1 ) {
		jLL = jL;
	}
	if (nvert[k][j+1][i]>0.1 ) {
	  //jR = j;
		jRR = jR;
	}
	if ( j!=my-2 && nvert[k][j+2][i]>1.1 ) {
		jRR = jR;
	}
	
	if(j==1) {
		if(j_periodic) jLL = my-3;
		else if(jj_periodic) jLL=-3;
		else jLL = jL;
	}
	else if(j==my-2) {
		if(j_periodic) jRR = 2;
		else if(jj_periodic) jRR=my+2;
		else jRR = jR;
	}
	
	dplus0=level[k][jR][i]-level[k][j][i], dminus0=level[k][j][i]-level[k][jL][i];
	
	_dplus  =  dplus0 - 0.5 * M(level[k][jR][i]-2.*level[k][j][i]+level[k][jL][i], level[k][jRR][i]-2.*level[k][jR][i]+level[k][j][i]) * f;
	_dminus = dminus0 + 0.5 * M(level[k][jR][i]-2.*level[k][j][i]+level[k][jL][i], level[k][j][i]-2.*level[k][jL][i]+level[k][jLL][i]) * f;
			
	if( sgn*dplus0<0 && sgn*dminus0<-sgn*dplus0) *dlde=_dplus;
	else if( sgn*dminus0>0 && sgn*dplus0>-sgn*dminus0) *dlde=_dminus;
	else if( sgn*dminus0<0 && sgn*dplus0>0) *dlde=0;
	else *dlde=0.5*(_dplus+_dminus);
	
	// k-direction
	int kLL=k-2, kL=k-1, kR=k+1, kRR=k+2;
	
	if (nvert[k-1][j][i]>0.1 ) {
	  //kL = k;
		kLL = kL;
	}
	if ( k!=1 && nvert[k-2][j][i]>1.1 ) {
		kLL = kL;
	}
	if (nvert[k+1][j][i]>0.1 ) {
	  //kR = k;
		kRR = kR;
	}
	if ( k!=mz-2 && nvert[k+2][j][i]>1.1 ) {
		kRR = kR;
	}
	
	if(k==1) {
		if(k_periodic) kLL = mz-3;
		else if(kk_periodic) kLL=-3;
		else kLL = kL;
	}
	else if(k==mz-2) {
		if(k_periodic) kRR = 2;
		else if(kk_periodic) kRR=mz+2;
		else kRR = kR;
	}
	
	dplus0=level[kR][j][i]-level[k][j][i], dminus0=level[k][j][i]-level[kL][j][i];
	
	_dplus  =  dplus0 - 0.5 * M(level[kR][j][i]-2.*level[k][j][i]+level[kL][j][i], level[kRR][j][i]-2.*level[kR][j][i]+level[k][j][i]) * f;
	_dminus = dminus0 + 0.5 * M(level[kR][j][i]-2.*level[k][j][i]+level[kL][j][i], level[k][j][i]-2.*level[kL][j][i]+level[kLL][j][i]) * f;
		
	if( sgn*dplus0<0 && sgn*dminus0<-sgn*dplus0) *dldz=_dplus;
	else if( sgn*dminus0>0 && sgn*dplus0>-sgn*dminus0) *dldz=_dminus;
	else if( sgn*dminus0<0 && sgn*dplus0>0) *dldz=0;
	else *dldz=0.5*(_dplus+_dminus);
}

