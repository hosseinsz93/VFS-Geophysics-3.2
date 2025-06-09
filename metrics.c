#include "variables.h"
#define NEWMETRIC

PetscErrorCode FormMetrics(UserCtx *user)
{
  DM		cda;
  Cmpnts	***csi, ***eta, ***zet;
  PetscScalar	***aj;
  Vec		coords;
  Cmpnts	***coor;

  DM		da = user->da, fda = user->fda;
  Vec		Csi = user->Csi, Eta = user->Eta, Zet = user->Zet;
  Vec		Aj = user->Aj;
  Vec		ICsi = user->ICsi, IEta = user->IEta, IZet = user->IZet;
  Vec		JCsi = user->JCsi, JEta = user->JEta, JZet = user->JZet;
  Vec		KCsi = user->KCsi, KEta = user->KEta, KZet = user->KZet;
  Vec		IAj = user->IAj, JAj = user->JAj, KAj = user->KAj;

  
  Cmpnts	***icsi, ***ieta, ***izet;
  Cmpnts	***jcsi, ***jeta, ***jzet;
  Cmpnts	***kcsi, ***keta, ***kzet;
  Cmpnts	***gs;
  PetscReal	***iaj, ***jaj, ***kaj;

  Vec		Cent = user->Cent; //local working array for storing cell center geometry

  Vec		Centx, Centy, Centz, lCoor;
  Cmpnts	***cent, ***centx, ***centy, ***centz;

  int	xs, ys, zs, xe, ye, ze;
  DMDALocalInfo	info;

  int	mx, my, mz;
  int	lxs, lxe, lys, lye, lzs, lze;

  int	i, j, k;
  int	gxs, gxe, gys, gye, gzs, gze;
  PetscErrorCode	ierr;

  PetscReal	xcp, ycp, zcp, xcm, ycm, zcm;
  DMDAGetLocalInfo(da, &info);
  mx = info.mx; my = info.my; mz = info.mz;
  xs = info.xs; xe = xs + info.xm;
  ys = info.ys; ye = ys + info.ym;
  zs = info.zs; ze = zs + info.zm;

  gxs = info.gxs; gxe = gxs + info.gxm;
  gys = info.gys; gye = gys + info.gym;
  gzs = info.gzs; gze = gzs + info.gzm;

  DMDAGetCoordinateDA(da, &cda);
  DMDAVecGetArray(fda, Csi, &csi);
  DMDAVecGetArray(fda, Eta, &eta);
  DMDAVecGetArray(fda, Zet, &zet);
  ierr = DMDAVecGetArray(da, Aj,  &aj); CHKERRQ(ierr);

  DMDAGetGhostedCoordinates(da, &coords);
  DMDAVecGetArray(fda, coords, &coor);

  DMGetLocalVector(fda, &Centx);

  VecDuplicate(Centx, &Centy);
  VecDuplicate(Centx, &Centz);

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;


	DMDAVecGetArray(fda, user->Cent, &cent);
  	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++) 
	for (i=lxs; i<lxe; i++) {
		cent[k][j][i].x = 0.125 *
			(coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
			coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x +
			coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x +
			coor[k-1][j  ][i-1].x + coor[k-1][j-1][i-1].x);
		cent[k][j][i].y = 0.125 *
			(coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
			coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y +
			coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y +
			coor[k-1][j  ][i-1].y + coor[k-1][j-1][i-1].y);
		cent[k][j][i].z = 0.125 *
			(coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
			coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z +
			coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z +
			coor[k-1][j  ][i-1].z + coor[k-1][j-1][i-1].z);
	}
	DMDAVecRestoreArray(fda, user->Cent, &cent);
	
	DMGlobalToLocalBegin(fda, user->Cent, INSERT_VALUES, user->lCent);
	DMGlobalToLocalEnd(fda, user->Cent, INSERT_VALUES, user->lCent);
	
	
	
	#if !defined (NEWMETRIC)
  /* Calculating transformation metrics in i direction */
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=xs; i<lxe; i++) {
		/* csi = de X dz */
		double dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
			      coor[k  ][j-1][i  ].x - coor[k-1][j-1][i  ].x);
		double dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
			      coor[k  ][j-1][i  ].y - coor[k-1][j-1][i  ].y);
		double dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
			      coor[k  ][j-1][i  ].z - coor[k-1][j-1][i  ].z);

		double dxdz = 0.5 * (coor[k  ][j-1][i  ].x + coor[k  ][j  ][i  ].x -
			      coor[k-1][j-1][i  ].x - coor[k-1][j  ][i  ].x);
		double dydz = 0.5 * (coor[k  ][j-1][i  ].y + coor[k  ][j  ][i  ].y -
			      coor[k-1][j-1][i  ].y - coor[k-1][j  ][i  ].y);
		double dzdz = 0.5 * (coor[k  ][j-1][i  ].z + coor[k  ][j  ][i  ].z -
			      coor[k-1][j-1][i  ].z - coor[k-1][j  ][i  ].z);
		
		csi[k][j][i].x = dyde * dzdz - dzde * dydz;
		csi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
		csi[k][j][i].z = dxde * dydz - dyde * dxdz;
		
	}
    
	// Need more work -- lg65
	/* calculating j direction metrics */
	for (k=lzs; k<lze; k++)
	for (j=ys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		/* eta = dz X de */
		double dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
			      coor[k  ][j  ][i-1].x - coor[k-1][j  ][i-1].x);
		double dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
			      coor[k  ][j  ][i-1].y - coor[k-1][j  ][i-1].y);
		double dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
			      coor[k  ][j  ][i-1].z - coor[k-1][j  ][i-1].z);
											 
		double dxdz = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
			      coor[k-1][j  ][i  ].x - coor[k-1][j  ][i-1].x);
		double dydz = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
			      coor[k-1][j  ][i  ].y - coor[k-1][j  ][i-1].y);
		double dzdz = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
			      coor[k-1][j  ][i  ].z - coor[k-1][j  ][i-1].z);
		
		eta[k][j][i].x = dydz * dzdc - dzdz * dydc;
		eta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
		eta[k][j][i].z = dxdz * dydc - dydz * dxdc;
	}

	/* calculating k direction metrics */
	for (k=zs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x -
			      coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x);
		double dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y -
			      coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y);
		double dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z -
			      coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z);
									 
		double dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
			      coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x);
		double dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
			      coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y);
		double dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
			      coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z);
		
		zet[k][j][i].x = dydc * dzde - dzdc * dyde;
		zet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
		zet[k][j][i].z = dxdc * dyde - dydc * dxde;
		
	}
	
	#else
	Cmpnts ***csitmp, ***etatmp, ***zettmp;
	Vec lCsitmp, lEtatmp, lZettmp;
	DMCreateLocalVector(user->fda, &lCsitmp);
	DMCreateLocalVector(user->fda, &lEtatmp);
	DMCreateLocalVector(user->fda, &lZettmp);
	
	DMDAVecGetArray(fda, lCsitmp, &csitmp);
	DMDAVecGetArray(fda, lEtatmp, &etatmp);
	DMDAVecGetArray(fda, lZettmp, &zettmp);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=xs; i<lxe; i++) {
		/* csi = de X dz */
		double dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
			      coor[k  ][j-1][i  ].x - coor[k-1][j-1][i  ].x);
		double dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
			      coor[k  ][j-1][i  ].y - coor[k-1][j-1][i  ].y);
		double dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
			      coor[k  ][j-1][i  ].z - coor[k-1][j-1][i  ].z);

		double dxdz = 0.5 * (coor[k  ][j-1][i  ].x + coor[k  ][j  ][i  ].x -
			      coor[k-1][j-1][i  ].x - coor[k-1][j  ][i  ].x);
		double dydz = 0.5 * (coor[k  ][j-1][i  ].y + coor[k  ][j  ][i  ].y -
			      coor[k-1][j-1][i  ].y - coor[k-1][j  ][i  ].y);
		double dzdz = 0.5 * (coor[k  ][j-1][i  ].z + coor[k  ][j  ][i  ].z -
			      coor[k-1][j-1][i  ].z - coor[k-1][j  ][i  ].z);
		
		csitmp[k][j][i].x = dyde * dzdz - dzde * dydz;
		csitmp[k][j][i].y =-dxde * dzdz + dzde * dxdz;
		csitmp[k][j][i].z = dxde * dydz - dyde * dxdz;
		
	}
    
	// Need more work -- lg65
	/* calculating j direction metrics */
	for (k=lzs; k<lze; k++)
	for (j=ys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		/* eta = dz X de */
		double dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
			      coor[k  ][j  ][i-1].x - coor[k-1][j  ][i-1].x);
		double dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
			      coor[k  ][j  ][i-1].y - coor[k-1][j  ][i-1].y);
		double dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
			      coor[k  ][j  ][i-1].z - coor[k-1][j  ][i-1].z);
											 
		double dxdz = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
			      coor[k-1][j  ][i  ].x - coor[k-1][j  ][i-1].x);
		double dydz = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
			      coor[k-1][j  ][i  ].y - coor[k-1][j  ][i-1].y);
		double dzdz = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
			      coor[k-1][j  ][i  ].z - coor[k-1][j  ][i-1].z);
		
		etatmp[k][j][i].x = dydz * dzdc - dzdz * dydc;
		etatmp[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
		etatmp[k][j][i].z = dxdz * dydc - dydz * dxdc;
		
		//if(i==1 && j==0 && (k==1 || k==33)) printf("(%02d,%02d,%02d) eta.x=%f eta.y=%f eta.z=%f\n", i,j,k, etatmp[k][j][i].x, etatmp[k][j][i].y, etatmp[k][j][i].z);
	}

	/* calculating k direction metrics */
	for (k=zs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x -
			      coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x);
		double dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y -
			      coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y);
		double dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z -
			      coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z);
									 
		double dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
			      coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x);
		double dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
			      coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y);
		double dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
			      coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z);
		
		zettmp[k][j][i].x = dydc * dzde - dzdc * dyde;
		zettmp[k][j][i].y =-dxdc * dzde + dzdc * dxde;
		zettmp[k][j][i].z = dxdc * dyde - dydc * dxde;
		
	}
	
	DMDAVecRestoreArray(fda, lCsitmp, &csitmp);
	DMDAVecRestoreArray(fda, lEtatmp, &etatmp);
	DMDAVecRestoreArray(fda, lZettmp, &zettmp);
	
	DMDALocalToLocalBegin(fda, lCsitmp, INSERT_VALUES, lCsitmp);
	DMDALocalToLocalEnd(fda, lCsitmp, INSERT_VALUES, lCsitmp);
	
	DMDALocalToLocalBegin(fda, lEtatmp, INSERT_VALUES, lEtatmp);
	DMDALocalToLocalEnd(fda, lEtatmp, INSERT_VALUES, lEtatmp);
	
	DMDALocalToLocalBegin(fda, lZettmp, INSERT_VALUES, lZettmp);
	DMDALocalToLocalEnd(fda, lZettmp, INSERT_VALUES, lZettmp);
	
	DMDAVecGetArray(fda, lCsitmp, &csitmp);
	DMDAVecGetArray(fda, lEtatmp, &etatmp);
	DMDAVecGetArray(fda, lZettmp, &zettmp);
	
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		AxByC ( 0.5, csitmp[k][j][i], 0.5, csitmp[k][j][i-1], &csi[k][j][i]);
		AxByC ( 0.5, etatmp[k][j][i], 0.5, etatmp[k][j-1][i], &eta[k][j][i]);
		AxByC ( 0.5, zettmp[k][j][i], 0.5, zettmp[k-1][j][i], &zet[k][j][i]);
		
		//if(i==1 && j==1 && (k==1 || k==33)) printf("(%02d,%02d,%02d) eta.x=%f eta.y=%f eta.z=%f\n", i,j,k, eta[k][j][i].x, eta[k][j][i].y, eta[k][j][i].z);
	}
	DMDAVecRestoreArray(fda, lCsitmp, &csitmp);
	DMDAVecRestoreArray(fda, lEtatmp, &etatmp);
	DMDAVecRestoreArray(fda, lZettmp, &zettmp);
	
	VecDestroy(&lCsitmp);
	VecDestroy(&lEtatmp);
	VecDestroy(&lZettmp);
	#endif
	
  /* calculating Jacobian of the transformation */
  for (k=lzs; k<lze; k++){
    for (j=lys; j<lye; j++){
      for (i=lxs; i<lxe; i++){

	if (i>0 && j>0 && k>0) {
	  double dxdc = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
			 coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x -
			 coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x -
			 coor[k-1][j  ][i-1].x - coor[k-1][j-1][i-1].x);
	  double dydc = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
			 coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y -
			 coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y -
			 coor[k-1][j  ][i-1].y - coor[k-1][j-1][i-1].y);
	  double dzdc = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
			 coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z -
			 coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z -
			 coor[k-1][j  ][i-1].z - coor[k-1][j-1][i-1].z);

	  double dxde = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x +
			 coor[k-1][j  ][i  ].x + coor[k-1][j  ][i-1].x - 
			 coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x -
			 coor[k-1][j-1][i  ].x - coor[k-1][j-1][i-1].x);
	  double dyde = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y +
			 coor[k-1][j  ][i  ].y + coor[k-1][j  ][i-1].y - 
			 coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y -
			 coor[k-1][j-1][i  ].y - coor[k-1][j-1][i-1].y);
	  double dzde = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z +
			 coor[k-1][j  ][i  ].z + coor[k-1][j  ][i-1].z - 
			 coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z -
			 coor[k-1][j-1][i  ].z - coor[k-1][j-1][i-1].z);

	  double dxdz = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
			 coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x -
			 coor[k-1][j  ][i  ].x - coor[k-1][j-1][i  ].x -
			 coor[k-1][j  ][i-1].x - coor[k-1][j-1][i-1].x);
	  double dydz = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
			 coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y -
			 coor[k-1][j  ][i  ].y - coor[k-1][j-1][i  ].y -
			 coor[k-1][j  ][i-1].y - coor[k-1][j-1][i-1].y);
	  double dzdz = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
			 coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z -
			 coor[k-1][j  ][i  ].z - coor[k-1][j-1][i  ].z -
			 coor[k-1][j  ][i-1].z - coor[k-1][j-1][i-1].z);
	  
		aj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
				dydc * (dxde * dzdz - dzde * dxdz) +
				dzdc * (dxde * dydz - dyde * dxdz);
		aj[k][j][i] = 1./aj[k][j][i];
	  /*
		#ifdef NEWMETRIC
		csi[k][j][i].x = dyde * dzdz - dzde * dydz;
		csi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
		csi[k][j][i].z = dxde * dydz - dyde * dxdz;
		
		eta[k][j][i].x = dydz * dzdc - dzdz * dydc;
		eta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
		eta[k][j][i].z = dxdz * dydc - dydz * dxdc;
	  
		zet[k][j][i].x = dydc * dzde - dzdc * dyde;
		zet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
		zet[k][j][i].z = dxdc * dyde - dydc * dxde;
		#endif
		*/
	}
      }
    }
  }

  // mirror grid outside the boundary
	if (xs==0) {
		i = xs;
		for (k=zs; k<ze; k++) 
		for (j=ys; j<ye; j++) {
			#ifdef NEWMETRIC
			csi[k][j][i] = csi[k][j][i+1];
			#endif
			eta[k][j][i] = eta[k][j][i+1];
			zet[k][j][i] = zet[k][j][i+1];
			aj[k][j][i] = aj[k][j][i+1];
		}
	}

	if (xe==mx) {
		i = xe-1;
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			#ifdef NEWMETRIC
			csi[k][j][i] = csi[k][j][i-1];
			#endif
			eta[k][j][i] = eta[k][j][i-1];
			zet[k][j][i] = zet[k][j][i-1];
			aj[k][j][i] = aj[k][j][i-1];
		}
	}
  

	if (ys==0) {
		j = ys;
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			eta[k][j][i] = eta[k][j+1][i];
			#endif
			csi[k][j][i] = csi[k][j+1][i];
			zet[k][j][i] = zet[k][j+1][i];
			aj[k][j][i] = aj[k][j+1][i];
		}
	}
  

	if (ye==my) {
		j = ye-1;
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			eta[k][j][i] = eta[k][j-1][i];
			#endif
			csi[k][j][i] = csi[k][j-1][i];
			zet[k][j][i] = zet[k][j-1][i];
			aj[k][j][i] = aj[k][j-1][i];
		}
	}
  

	if (zs==0) {
		k = zs;
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			zet[k][j][i] = zet[k+1][j][i];
			#endif
			eta[k][j][i] = eta[k+1][j][i];
			csi[k][j][i] = csi[k+1][j][i];
			aj[k][j][i] = aj[k+1][j][i];
		}
	}
	

	if (ze==mz) {
		k = ze-1;
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			zet[k][j][i] = zet[k-1][j][i];
			#endif
			eta[k][j][i] = eta[k-1][j][i];
			csi[k][j][i] = csi[k-1][j][i];
			aj[k][j][i] = aj[k-1][j][i];
		}
	}
	
	DMDAVecRestoreArray(fda, Csi, &csi);
	DMDAVecRestoreArray(fda, Eta, &eta);
	DMDAVecRestoreArray(fda, Zet, &zet);
	DMDAVecRestoreArray(da, Aj,  &aj);
	
	DMGlobalToLocalBegin(fda, user->Csi, INSERT_VALUES, user->lCsi);
	DMGlobalToLocalEnd(fda, user->Csi, INSERT_VALUES, user->lCsi);

	DMGlobalToLocalBegin(fda, user->Eta, INSERT_VALUES, user->lEta);
	DMGlobalToLocalEnd(fda, user->Eta, INSERT_VALUES, user->lEta);

	DMGlobalToLocalBegin(fda, user->Zet, INSERT_VALUES, user->lZet);
	DMGlobalToLocalEnd(fda, user->Zet, INSERT_VALUES, user->lZet);
  
	DMGlobalToLocalBegin(da, user->Aj, INSERT_VALUES, user->lAj);
	DMGlobalToLocalEnd(da, user->Aj, INSERT_VALUES, user->lAj);
	
		DMDAVecGetArray(fda, user->lCsi, &csi);
		DMDAVecGetArray(fda, user->lEta, &eta);
		DMDAVecGetArray(fda, user->lZet, &zet);
		DMDAVecGetArray(da, user->lAj,  &aj);
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			int flag=0, a=i, b=j, c=k;
			int i_flag=0, j_flag=0, k_flag=0;
			
			if(i_periodic && i==0) a=mx-2, i_flag=1;
			else if(i_periodic && i==mx-1) a=1, i_flag=1;
			
			if(j_periodic && j==0) b=my-2, j_flag=1;
			else if(j_periodic && j==my-1) b=1, j_flag=1;
			
			if(k_periodic && k==0) c=mz-2, k_flag=1;
			else if(k_periodic && k==mz-1) c=1, k_flag=1;
			
			if(ii_periodic && i==0) a=-2, i_flag=1;
			else if(ii_periodic && i==mx-1) a=mx+1, i_flag=1;
			
			if(jj_periodic && j==0) b=-2, j_flag=1;
			else if(jj_periodic && j==my-1) b=my+1, j_flag=1;
			
			if(kk_periodic && k==0) c=-2, k_flag=1;
			else if(kk_periodic && k==mz-1) c=mz+1, k_flag=1;
			
			flag = i_flag + j_flag + k_flag;

			if(flag) {
				csi[k][j][i] = csi[c][b][a];
				eta[k][j][i] = eta[c][b][a];
				zet[k][j][i] = zet[c][b][a];
				aj[k][j][i] = aj[c][b][a];
			}
		}
		DMDAVecRestoreArray(fda, user->lCsi, &csi);
		DMDAVecRestoreArray(fda, user->lEta, &eta);
		DMDAVecRestoreArray(fda, user->lZet, &zet);
		DMDAVecRestoreArray(da, user->lAj,  &aj);
	
	Cmpnts	***lcsi, ***leta, ***lzet;
	PetscScalar	***laj;
	
	
	DMDAVecGetArray(fda, user->lCsi, &lcsi);
	DMDAVecGetArray(fda, user->lEta, &leta);
	DMDAVecGetArray(fda, user->lZet, &lzet);
	DMDAVecGetArray(da, user->lAj,  &laj);
	
  DMDAVecGetArray(fda, ICsi, &icsi);
  DMDAVecGetArray(fda, IEta, &ieta);
  DMDAVecGetArray(fda, IZet, &izet);
  DMDAVecGetArray(da, IAj,  &iaj);

	

	DMDAVecGetArray(fda, user->GridSpace, &gs);	
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	xcp = 0.25 *
	  (coor[k][j][i].x + coor[k][j-1][i].x +
	   coor[k-1][j-1][i].x + coor[k-1][j][i].x);
	ycp = 0.25 *
	  (coor[k][j][i].y + coor[k][j-1][i].y +
	   coor[k-1][j-1][i].y + coor[k-1][j][i].y);
	zcp = 0.25 *
	  (coor[k][j][i].z + coor[k][j-1][i].z +
	   coor[k-1][j-1][i].z + coor[k-1][j][i].z);

	xcm = 0.25 *
	  (coor[k][j][i-1].x + coor[k][j-1][i-1].x +
	   coor[k-1][j-1][i-1].x + coor[k-1][j][i-1].x);
	ycm = 0.25 *
	  (coor[k][j][i-1].y + coor[k][j-1][i-1].y +
	   coor[k-1][j-1][i-1].y + coor[k-1][j][i-1].y);
	zcm = 0.25 *
	  (coor[k][j][i-1].z + coor[k][j-1][i-1].z +
	   coor[k-1][j-1][i-1].z + coor[k-1][j][i-1].z);

	gs[k][j][i].x = sqrt((xcp-xcm) * (xcp-xcm) +
			     (ycp-ycm) * (ycp-ycm) +
			     (zcp-zcm) * (zcp-zcm));

	xcp = 0.25 *
	  (coor[k][j][i].x + coor[k][j][i-1].x +
	   coor[k-1][j][i].x + coor[k-1][j][i-1].x);
	ycp = 0.25 *
	  (coor[k][j][i].y + coor[k][j][i-1].y +
	   coor[k-1][j][i].y + coor[k-1][j][i-1].y);
	zcp = 0.25 *
	  (coor[k][j][i].z + coor[k][j][i-1].z +
	   coor[k-1][j][i].z + coor[k-1][j][i-1].z);

	xcm = 0.25 *
	  (coor[k][j-1][i].x + coor[k][j-1][i-1].x +
	   coor[k-1][j-1][i].x + coor[k-1][j-1][i-1].x);
	ycm = 0.25 *
	  (coor[k][j-1][i].y + coor[k][j-1][i-1].y +
	   coor[k-1][j-1][i].y + coor[k-1][j-1][i-1].y);
	zcm = 0.25 *
	  (coor[k][j-1][i].z + coor[k][j-1][i-1].z +
	   coor[k-1][j-1][i].z + coor[k-1][j-1][i-1].z);

	gs[k][j][i].y = sqrt((xcp-xcm) * (xcp-xcm) +
			     (ycp-ycm) * (ycp-ycm) +
			     (zcp-zcm) * (zcp-zcm));

	xcp = 0.25 *
	  (coor[k][j][i].x + coor[k][j][i-1].x +
	   coor[k][j-1][i].x + coor[k][j-1][i-1].x);
	ycp = 0.25 *
	  (coor[k][j][i].y + coor[k][j][i-1].y +
	   coor[k][j-1][i].y + coor[k][j-1][i-1].y);
	zcp = 0.25 *
	  (coor[k][j][i].z + coor[k][j][i-1].z +
	   coor[k][j-1][i].z + coor[k][j-1][i-1].z);

	xcm = 0.25 *
	  (coor[k-1][j][i].x + coor[k-1][j][i-1].x +
	   coor[k-1][j-1][i].x + coor[k-1][j-1][i-1].x);
	ycm = 0.25 *
	  (coor[k-1][j][i].y + coor[k-1][j][i-1].y +
	   coor[k-1][j-1][i].y + coor[k-1][j-1][i-1].y);
	zcm = 0.25 *
	  (coor[k-1][j][i].z + coor[k-1][j][i-1].z +
	   coor[k-1][j-1][i].z + coor[k-1][j-1][i-1].z);

	gs[k][j][i].z = sqrt((xcp-xcm) * (xcp-xcm) +
			     (ycp-ycm) * (ycp-ycm) +
			     (zcp-zcm) * (zcp-zcm));

      }
    }
  }

  DMDAVecRestoreArray(fda, user->GridSpace, &gs);

  DMDAVecGetArray(fda, Centx, &centx);
  for(k=gzs+1; k<gze; k++) {
    for (j=gys+1; j<gye; j++) {
      for (i=gxs; i<gxe; i++) {
	centx[k][j][i].x = (coor[k  ][j  ][i].x + coor[k-1][j  ][i].x +
			    coor[k  ][j-1][i].x + coor[k-1][j-1][i].x) * 0.25;
	centx[k][j][i].y = (coor[k  ][j  ][i].y + coor[k-1][j  ][i].y +
			    coor[k  ][j-1][i].y + coor[k-1][j-1][i].y) * 0.25;
	centx[k][j][i].z = (coor[k  ][j  ][i].z + coor[k-1][j  ][i].z +
			    coor[k  ][j-1][i].z + coor[k-1][j-1][i].z) * 0.25;

      }
    }
  }


  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<lxe; i++) {  
	PetscScalar	dxdc, dydc, dzdc, dxde, dyde, dzde, dxdz, dydz, dzdz;
	
	if (i==0) {
	  dxdc = centx[k][j][i+1].x - centx[k][j][i].x;
	  dydc = centx[k][j][i+1].y - centx[k][j][i].y;
	  dzdc = centx[k][j][i+1].z - centx[k][j][i].z;
	}
	else if (i==mx-2) {
		if(i_periodic) {
			dxdc = (centx[k][j][1].x - centx[k][j][0].x + centx[k][j][i].x - centx[k][j][i-1].x) * 0.5;
			dydc = (centx[k][j][1].y - centx[k][j][0].y + centx[k][j][i].y - centx[k][j][i-1].y) * 0.5;
			dzdc = (centx[k][j][1].z - centx[k][j][0].z + centx[k][j][i].z - centx[k][j][i-1].z) * 0.5;
		}
		else if(ii_periodic) {
			dxdc = (centx[k][j][mx+1].x - centx[k][j][mx+0].x + centx[k][j][i].x - centx[k][j][i-1].x) * 0.5;
			dydc = (centx[k][j][mx+1].y - centx[k][j][mx+0].y + centx[k][j][i].y - centx[k][j][i-1].y) * 0.5;
			dzdc = (centx[k][j][mx+1].z - centx[k][j][mx+0].z + centx[k][j][i].z - centx[k][j][i-1].z) * 0.5;
		}
		else {
			dxdc = centx[k][j][i].x - centx[k][j][i-1].x;
			dydc = centx[k][j][i].y - centx[k][j][i-1].y;
			dzdc = centx[k][j][i].z - centx[k][j][i-1].z;
		}
	}
	else {
	  dxdc = (centx[k][j][i+1].x - centx[k][j][i-1].x) * 0.5;
	  dydc = (centx[k][j][i+1].y - centx[k][j][i-1].y) * 0.5;
	  dzdc = (centx[k][j][i+1].z - centx[k][j][i-1].z) * 0.5;
	}

	
	if (j==1) {
	  dxde = centx[k][j+1][i].x - centx[k][j][i].x;
	  dyde = centx[k][j+1][i].y - centx[k][j][i].y;
	  dzde = centx[k][j+1][i].z - centx[k][j][i].z;
	}
	else if (j==my-2) {
	  dxde = centx[k][j][i].x - centx[k][j-1][i].x;
	  dyde = centx[k][j][i].y - centx[k][j-1][i].y;
	  dzde = centx[k][j][i].z - centx[k][j-1][i].z;
	}
	else {
	  dxde = (centx[k][j+1][i].x - centx[k][j-1][i].x) * 0.5;
	  dyde = (centx[k][j+1][i].y - centx[k][j-1][i].y) * 0.5;
	  dzde = (centx[k][j+1][i].z - centx[k][j-1][i].z) * 0.5;
	}
	
	if (k==1) {
	  dxdz = (centx[k+1][j][i].x - centx[k][j][i].x);
	  dydz = (centx[k+1][j][i].y - centx[k][j][i].y);
	  dzdz = (centx[k+1][j][i].z - centx[k][j][i].z);
	}
	else if (k==mz-2) {
	  dxdz = (centx[k][j][i].x - centx[k-1][j][i].x);
	  dydz = (centx[k][j][i].y - centx[k-1][j][i].y);
	  dzdz = (centx[k][j][i].z - centx[k-1][j][i].z);
	}
	else {
	  dxdz = (centx[k+1][j][i].x - centx[k-1][j][i].x) * 0.5;
	  dydz = (centx[k+1][j][i].y - centx[k-1][j][i].y) * 0.5;
	  dzdz = (centx[k+1][j][i].z - centx[k-1][j][i].z) * 0.5;
	}
	
	icsi[k][j][i].x = dyde * dzdz - dzde * dydz;
	icsi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
	icsi[k][j][i].z = dxde * dydz - dyde * dxdz;

	ieta[k][j][i].x = dydz * dzdc - dzdz * dydc;
	ieta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
	ieta[k][j][i].z = dxdz * dydc - dydz * dxdc;

	izet[k][j][i].x = dydc * dzde - dzdc * dyde;
	izet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
	izet[k][j][i].z = dxdc * dyde - dydc * dxde;

	iaj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz)
                     - dydc * (dxde * dzdz - dzde * dxdz)
                     + dzdc * (dxde * dydz - dyde * dxdz);
	iaj[k][j][i] = 1./iaj[k][j][i];
	
/*
	#ifdef NEWMETRIC
	if( (i_periodic || ii_periodic) && i==mx-2 ) {
		AxByC ( 0.5, lcsi[k][j][i], 0.5, lcsi[k][j][i+1], &icsi[k][j][i]);
		AxByC ( 0.5, leta[k][j][i], 0.5, leta[k][j][i+1], &ieta[k][j][i]);
		AxByC ( 0.5, lzet[k][j][i], 0.5, lzet[k][j][i+1], &izet[k][j][i]);
		iaj[k][j][i] = 2. / ( 1./laj[k][j][i] + 1./laj[k][j][i+1] );
	}
	#endif
*/
      }
    }
  }

  // PetscPrintf(PETSC_COMM_WORLD, "test\n");

  DMDAVecRestoreArray(fda, ICsi, &icsi);
  DMDAVecRestoreArray(fda, IEta, &ieta);
  DMDAVecRestoreArray(fda, IZet, &izet);
  DMDAVecRestoreArray(da, IAj,  &iaj);

  // j direction
  DMDAVecGetArray(fda, JCsi, &jcsi);
  DMDAVecGetArray(fda, JEta, &jeta);
  DMDAVecGetArray(fda, JZet, &jzet);
  DMDAVecGetArray(da, JAj,  &jaj);

  DMDAVecGetArray(fda, Centy, &centy);
  for(k=gzs+1; k<gze; k++) {
    for (j=gys; j<gye; j++) {
      for (i=gxs+1; i<gxe; i++) {
	centy[k][j][i].x = (coor[k  ][j][i  ].x + coor[k-1][j][i  ].x +
			    coor[k  ][j][i-1].x + coor[k-1][j][i-1].x) * 0.25;
	centy[k][j][i].y = (coor[k  ][j][i  ].y + coor[k-1][j][i  ].y +
			    coor[k  ][j][i-1].y + coor[k-1][j][i-1].y) * 0.25;
	centy[k][j][i].z = (coor[k  ][j][i  ].z + coor[k-1][j][i  ].z +
			    coor[k  ][j][i-1].z + coor[k-1][j][i-1].z) * 0.25;
      }
    }
  }

  for (k=lzs; k<lze; k++) {
    for (j=ys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	PetscScalar	dxdc, dydc, dzdc, dxde, dyde, dzde, dxdz, dydz, dzdz;
	if (i==1) {
	  dxdc = centy[k][j][i+1].x - centy[k][j][i].x;
	  dydc = centy[k][j][i+1].y - centy[k][j][i].y;
	  dzdc = centy[k][j][i+1].z - centy[k][j][i].z;
	}
	else if (i==mx-2) {
	  dxdc = centy[k][j][i].x - centy[k][j][i-1].x;
	  dydc = centy[k][j][i].y - centy[k][j][i-1].y;
	  dzdc = centy[k][j][i].z - centy[k][j][i-1].z;
	}
	else {
	  dxdc = (centy[k][j][i+1].x - centy[k][j][i-1].x) * 0.5;
	  dydc = (centy[k][j][i+1].y - centy[k][j][i-1].y) * 0.5;
	  dzdc = (centy[k][j][i+1].z - centy[k][j][i-1].z) * 0.5;
	}

	if (j==0) {
	  dxde = centy[k][j+1][i].x - centy[k][j][i].x;
	  dyde = centy[k][j+1][i].y - centy[k][j][i].y;
	  dzde = centy[k][j+1][i].z - centy[k][j][i].z;
	}
	else if (j==my-2) {
		if(j_periodic) {
			dxde = (centy[k][1][i].x - centy[k][0][i].x + centy[k][j][i].x - centy[k][j-1][i].x) * 0.5;
			dyde = (centy[k][1][i].y - centy[k][0][i].y + centy[k][j][i].y - centy[k][j-1][i].y) * 0.5;
			dzde = (centy[k][1][i].z - centy[k][0][i].z + centy[k][j][i].z - centy[k][j-1][i].z) * 0.5;
		}
		else if(jj_periodic) {
			dxde = (centy[k][my+1][i].x - centy[k][my+0][i].x + centy[k][j][i].x - centy[k][j-1][i].x) * 0.5;
			dyde = (centy[k][my+1][i].y - centy[k][my+0][i].y + centy[k][j][i].y - centy[k][j-1][i].y) * 0.5;
			dzde = (centy[k][my+1][i].z - centy[k][my+0][i].z + centy[k][j][i].z - centy[k][j-1][i].z) * 0.5;
		}
		else {
			dxde = centy[k][j][i].x - centy[k][j-1][i].x;
			dyde = centy[k][j][i].y - centy[k][j-1][i].y;
			dzde = centy[k][j][i].z - centy[k][j-1][i].z;
		}
	}
	else {
	  dxde = (centy[k][j+1][i].x - centy[k][j-1][i].x) * 0.5;
	  dyde = (centy[k][j+1][i].y - centy[k][j-1][i].y) * 0.5;
	  dzde = (centy[k][j+1][i].z - centy[k][j-1][i].z) * 0.5;
	}

	if (k==1) {
	  dxdz = (centy[k+1][j][i].x - centy[k][j][i].x);
	  dydz = (centy[k+1][j][i].y - centy[k][j][i].y);
	  dzdz = (centy[k+1][j][i].z - centy[k][j][i].z);
	}
	else if (k==mz-2) {
	  dxdz = (centy[k][j][i].x - centy[k-1][j][i].x);
	  dydz = (centy[k][j][i].y - centy[k-1][j][i].y);
	  dzdz = (centy[k][j][i].z - centy[k-1][j][i].z);
	}
	else {
	  dxdz = (centy[k+1][j][i].x - centy[k-1][j][i].x) * 0.5;
	  dydz = (centy[k+1][j][i].y - centy[k-1][j][i].y) * 0.5;
	  dzdz = (centy[k+1][j][i].z - centy[k-1][j][i].z) * 0.5;
	}

	jcsi[k][j][i].x = dyde * dzdz - dzde * dydz;
	jcsi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
	jcsi[k][j][i].z = dxde * dydz - dyde * dxdz;
	
	jeta[k][j][i].x = dydz * dzdc - dzdz * dydc;
	jeta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
	jeta[k][j][i].z = dxdz * dydc - dydz * dxdc;

	jzet[k][j][i].x = dydc * dzde - dzdc * dyde;
	jzet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
	jzet[k][j][i].z = dxdc * dyde - dydc * dxde;

	
	jaj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
	  dydc * (dxde * dzdz - dzde * dxdz) +
	  dzdc * (dxde * dydz - dyde * dxdz);
	jaj[k][j][i] = 1./jaj[k][j][i];
/*
	#ifdef NEWMETRIC
	if( (j_periodic || jj_periodic) && j==my-2 ) {
		AxByC ( 0.5, lcsi[k][j][i], 0.5, lcsi[k][j+1][i], &jcsi[k][j][i]);
		AxByC ( 0.5, leta[k][j][i], 0.5, leta[k][j+1][i], &jeta[k][j][i]);
		AxByC ( 0.5, lzet[k][j][i], 0.5, lzet[k][j+1][i], &jzet[k][j][i]);
		jaj[k][j][i] = 2. / ( 1./laj[k][j][i] + 1./laj[k][j+1][i] );
	}
	#endif
	*/
      }
    }
  }

  DMDAVecRestoreArray(fda, JCsi, &jcsi);
  DMDAVecRestoreArray(fda, JEta, &jeta);
  DMDAVecRestoreArray(fda, JZet, &jzet);
  DMDAVecRestoreArray(da, JAj,  &jaj);

  // k direction
  DMDAVecGetArray(fda, KCsi, &kcsi);
  DMDAVecGetArray(fda, KEta, &keta);
  DMDAVecGetArray(fda, KZet, &kzet);
  DMDAVecGetArray(da, KAj,  &kaj);

  DMDAVecGetArray(fda, Centz, &centz);
  for(k=gzs; k<gze; k++) {
    for (j=gys+1; j<gye; j++) {
      for (i=gxs+1; i<gxe; i++) {
	centz[k][j][i].x = (coor[k  ][j][i  ].x + coor[k][j-1][i  ].x +
			    coor[k  ][j][i-1].x + coor[k][j-1][i-1].x) * 0.25;
	centz[k][j][i].y = (coor[k  ][j][i  ].y + coor[k][j-1][i  ].y +
			    coor[k  ][j][i-1].y + coor[k][j-1][i-1].y) * 0.25;
	centz[k][j][i].z = (coor[k  ][j][i  ].z + coor[k][j-1][i  ].z +
			    coor[k  ][j][i-1].z + coor[k][j-1][i-1].z) * 0.25;
      }
    }
  }

  for (k=zs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {  
	PetscScalar	dxdc, dydc, dzdc, dxde, dyde, dzde, dxdz, dydz, dzdz;
	
	if (i==1) {
		dxdc = centz[k][j][i+1].x - centz[k][j][i].x;
		dydc = centz[k][j][i+1].y - centz[k][j][i].y;
		dzdc = centz[k][j][i+1].z - centz[k][j][i].z;
	}
	else if (i==mx-2) {
		/*
		if(i_periodic) {
			dxdc = (centz[k][j][1].x - centz[k][j][0].x + centz[k][j][i].x - centz[k][j][i-1].x) * 0.5;
			dydc = (centz[k][j][1].y - centz[k][j][0].y + centz[k][j][i].y - centz[k][j][i-1].y) * 0.5;
			dzdc = (centz[k][j][1].z - centz[k][j][0].z + centz[k][j][i].z - centz[k][j][i-1].z) * 0.5;
		}
		else if(ii_periodic) {
			dxdc = (centz[k][j][mx+1].x - centz[k][j][mx+0].x + centz[k][j][i].x - centz[k][j][i-1].x) * 0.5;
			dydc = (centz[k][j][mx+1].y - centz[k][j][mx+0].y + centz[k][j][i].y - centz[k][j][i-1].y) * 0.5;
			dzdc = (centz[k][j][mx+1].z - centz[k][j][mx+0].z + centz[k][j][i].z - centz[k][j][i-1].z) * 0.5;
		}
		else*/ {
			dxdc = centz[k][j][i].x - centz[k][j][i-1].x;
			dydc = centz[k][j][i].y - centz[k][j][i-1].y;
			dzdc = centz[k][j][i].z - centz[k][j][i-1].z;
		}
	}
	else {
		dxdc = (centz[k][j][i+1].x - centz[k][j][i-1].x) * 0.5;
		dydc = (centz[k][j][i+1].y - centz[k][j][i-1].y) * 0.5;
		dzdc = (centz[k][j][i+1].z - centz[k][j][i-1].z) * 0.5;
	}

	
	if (j==1) {
		dxde = centz[k][j+1][i].x - centz[k][j][i].x;
		dyde = centz[k][j+1][i].y - centz[k][j][i].y;
		dzde = centz[k][j+1][i].z - centz[k][j][i].z;
	}
	else if (j==my-2) {
		/*
		
		if(j_periodic) {
			dxde = (centz[k][1][i].x - centz[k][0][i].x + centz[k][j][i].x - centz[k][j-1][i].x) * 0.5;
			dyde = (centz[k][1][i].y - centz[k][0][i].y + centz[k][j][i].y - centz[k][j-1][i].y) * 0.5;
			dzde = (centz[k][1][i].z - centz[k][0][i].z + centz[k][j][i].z - centz[k][j-1][i].z) * 0.5;
		}
		else if(jj_periodic) {
			dxde = (centz[k][my+1][i].x - centz[k][my+0][i].x + centz[k][j][i].x - centz[k][j-1][i].x) * 0.5;
			dyde = (centz[k][my+1][i].y - centz[k][my+0][i].y + centz[k][j][i].y - centz[k][j-1][i].y) * 0.5;
			dzde = (centz[k][my+1][i].z - centz[k][my+0][i].z + centz[k][j][i].z - centz[k][j-1][i].z) * 0.5;
		}
		else*/ {
			dxde = centz[k][j][i].x - centz[k][j-1][i].x;
			dyde = centz[k][j][i].y - centz[k][j-1][i].y;
			dzde = centz[k][j][i].z - centz[k][j-1][i].z;
		}
	}
	else {
		dxde = (centz[k][j+1][i].x - centz[k][j-1][i].x) * 0.5;
		dyde = (centz[k][j+1][i].y - centz[k][j-1][i].y) * 0.5;
		dzde = (centz[k][j+1][i].z - centz[k][j-1][i].z) * 0.5;
	}

	
	if (k==0) {
		dxdz = (centz[k+1][j][i].x - centz[k][j][i].x);
		dydz = (centz[k+1][j][i].y - centz[k][j][i].y);
		dzdz = (centz[k+1][j][i].z - centz[k][j][i].z);
	}
	else if (k==mz-2) {
		if(k_periodic) {
			dxdz = (centz[1][j][i].x - centz[0][j][i].x + centz[k][j][i].x - centz[k-1][j][i].x) * 0.5;
			dydz = (centz[1][j][i].y - centz[0][j][i].y + centz[k][j][i].y - centz[k-1][j][i].y) * 0.5;
			dzdz = (centz[1][j][i].z - centz[0][j][i].z + centz[k][j][i].z - centz[k-1][j][i].z) * 0.5;
		}
		else if(kk_periodic) {
			dxdz = (centz[mz+1][j][i].x - centz[mz+0][j][i].x + centz[k][j][i].x - centz[k-1][j][i].x) * 0.5;
			dydz = (centz[mz+1][j][i].y - centz[mz+0][j][i].y + centz[k][j][i].y - centz[k-1][j][i].y) * 0.5;
			dzdz = (centz[mz+1][j][i].z - centz[mz+0][j][i].z + centz[k][j][i].z - centz[k-1][j][i].z) * 0.5;
		}
		else {
			dxdz = (centz[k][j][i].x - centz[k-1][j][i].x);
			dydz = (centz[k][j][i].y - centz[k-1][j][i].y);
			dzdz = (centz[k][j][i].z - centz[k-1][j][i].z);
		}
	}
	else {
		dxdz = (centz[k+1][j][i].x - centz[k-1][j][i].x) * 0.5;
		dydz = (centz[k+1][j][i].y - centz[k-1][j][i].y) * 0.5;
		dzdz = (centz[k+1][j][i].z - centz[k-1][j][i].z) * 0.5;
	}

	kcsi[k][j][i].x = dyde * dzdz - dzde * dydz;
	kcsi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
	kcsi[k][j][i].z = dxde * dydz - dyde * dxdz;

	keta[k][j][i].x = dydz * dzdc - dzdz * dydc;
	keta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
	keta[k][j][i].z = dxdz * dydc - dydz * dxdc;

	kzet[k][j][i].x = dydc * dzde - dzdc * dyde;
	kzet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
	kzet[k][j][i].z = dxdc * dyde - dydc * dxde;


	kaj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
	  dydc * (dxde * dzdz - dzde * dxdz) +
	  dzdc * (dxde * dydz - dyde * dxdz);
	kaj[k][j][i] = 1./kaj[k][j][i];
	/*
	if( i==1 && j==1 && (k==0 || k==mz/3 || k==mz-2)) {
		printf("%d,%d,%d => kcsi.x=%f, kcsi.y=%f, kcsi.z=%f\n", i, j, k, kcsi[k][j][i].x, kcsi[k][j][i].y, kcsi[k][j][i].z);
		printf("%d,%d,%d => keta.x=%f, keta.y=%f, keta.z=%f\n", i, j, k, keta[k][j][i].x, keta[k][j][i].y, keta[k][j][i].z);
		printf("%d,%d,%d => kzet.x=%f, kzet.y=%f, kzet.z=%f\n", i, j, k, kzet[k][j][i].x, kzet[k][j][i].y, kzet[k][j][i].z);
		printf("%d,%d,%d => kaj=%f, aj=%f, %f\n***\n", i, j, k, kaj[k][j][i], laj[k][j][i], laj[k+1][j][i]);
	}
	*/	
		/*
	#ifdef NEWMETRIC
	if( (k_periodic || kk_periodic) && k==mz-2 ) {
		AxByC ( 0.5, lcsi[k][j][i], 0.5, lcsi[k+1][j][i], &kcsi[k][j][i]);
		AxByC ( 0.5, leta[k][j][i], 0.5, leta[k+1][j][i], &keta[k][j][i]);
		AxByC ( 0.5, lzet[k][j][i], 0.5, lzet[k+1][j][i], &kzet[k][j][i]);
		kaj[k][j][i] = 2. / ( 1/laj[k][j][i] + 1/laj[k+1][j][i] );
	}
	#endif
*/
      }
    }
  }
 // exit(0);
	DMDAVecRestoreArray(fda, user->lCsi, &lcsi);
	DMDAVecRestoreArray(fda, user->lEta, &leta);
	DMDAVecRestoreArray(fda, user->lZet, &lzet);
	DMDAVecRestoreArray(da, user->lAj,  &laj);

/* ==================================================================================             */
/*  Calculating the Area of each side of the grid */
/*  the numbering is the same as flux (Ucont) */
  
  /*
  PetscReal  x13,x24,y13,y24,z13,z24;
  PetscReal  Ar1,Ar2,Ar3;
  Cmpnts     ***area;
  
  DMDAVecGetArray(fda, user->Area, &area);
  
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<lxe; i++) {
	x13= coor[k][j][i].x-coor[k-1][j-1][i].x;
	y13= coor[k][j][i].y-coor[k-1][j-1][i].y;
	z13= coor[k][j][i].z-coor[k-1][j-1][i].z;

	x24= coor[k][j-1][i].x-coor[k-1][j][i].x;
	y24= coor[k][j-1][i].y-coor[k-1][j][i].y;
	z24= coor[k][j-1][i].z-coor[k-1][j][i].z;

	Ar1 =  y13*z24 - z13*y24 ;
	Ar2 =-(x13*z24 - z13*x24);
	Ar3 =  x13*y24 - y13*x24 ;
	
	area[k][j][i].x = 0.5*sqrt( Ar1*Ar1+Ar2*Ar2+Ar3*Ar3 );
	      
      }
    }
  }

  for (k=lzs; k<lze; k++) {
    for (j=ys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	x13= coor[k][j][i].x-coor[k-1][j][i-1].x;
	y13= coor[k][j][i].y-coor[k-1][j][i-1].y;
	z13= coor[k][j][i].z-coor[k-1][j][i-1].z;

	x24= coor[k][j][i-1].x-coor[k-1][j][i].x;
	y24= coor[k][j][i-1].y-coor[k-1][j][i].y;
	z24= coor[k][j][i-1].z-coor[k-1][j][i].z;

	Ar1 =  y13*z24 - z13*y24 ;
	Ar2 =-(x13*z24 - z13*x24);
	Ar3 =  x13*y24 - y13*x24 ;

	area[k][j][i].y = 0.5*sqrt( Ar1*Ar1+Ar2*Ar2+Ar3*Ar3 );
	
      }
    }
  }

  for (k=zs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	x13= coor[k][j][i].x-coor[k][j-1][i-1].x;
	y13= coor[k][j][i].y-coor[k][j-1][i-1].y;
	z13= coor[k][j][i].z-coor[k][j-1][i-1].z;

	x24= coor[k][j-1][i].x-coor[k][j][i-1].x;
	y24= coor[k][j-1][i].y-coor[k][j][i-1].y;
	z24= coor[k][j-1][i].z-coor[k][j][i-1].z;

	Ar1 =  y13*z24 - z13*y24 ;
	Ar2 =-(x13*z24 - z13*x24);
	Ar3 =  x13*y24 - y13*x24 ;

	area[k][j][i].z = 0.5*sqrt( Ar1*Ar1+Ar2*Ar2+Ar3*Ar3 );
	
      }
    }
  }

  DMDAVecRestoreArray(fda, user->Area, &area);

  DMGlobalToLocalBegin(fda, user->Area, INSERT_VALUES, user->lArea);
  DMGlobalToLocalEnd(fda, user->Area, INSERT_VALUES, user->lArea);

  VecDestroy(&user->Area);
*/

/* ==================================================================================             */
/*  Calculating the Volume of each grid cell*/
/*  the numbering is the same as cell center (nvert, ucat) */
/*
  Cmpnts      v1,v2,v3;
  PetscReal   ***vol;

  DMDAVecGetArray(da, user->Volume, &vol);

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	
	v1.x = centx[k][j][i].x - centx[k][j][i-1].x;
	v1.y = centx[k][j][i].y - centx[k][j][i-1].y;
	v1.z = centx[k][j][i].z - centx[k][j][i-1].z;

	v2.x = centy[k][j][i].x - centy[k][j-1][i].x;
	v2.y = centy[k][j][i].y - centy[k][j-1][i].y;
	v2.z = centy[k][j][i].z - centy[k][j-1][i].z;

	v3.x = centz[k][j][i].x - centz[k-1][j][i].x;
	v3.y = centz[k][j][i].y - centz[k-1][j][i].y;
	v3.z = centz[k][j][i].z - centz[k-1][j][i].z;

	// Volume = v1.(v2xv3)
	vol[k][j][i] = v1.x*(v2.y*v3.z-v3.y*v2.z)
	              -v1.y*(v2.x*v3.z-v3.x*v2.z) +
           	       v1.z*(v2.x*v3.y-v3.x*v2.y);

	vol[k][j][i] = fabs(vol[k][j][i]);

      }
    }
  }

  DMDAVecRestoreArray(da, user->Volume, &vol);

  DMGlobalToLocalBegin(da, user->Volume, INSERT_VALUES, user->lVolume);
  DMGlobalToLocalEnd(da, user->Volume, INSERT_VALUES, user->lVolume);

  VecDestroy(&user->Volume);
  
  */
  
/* ==================================================================================             */

  DMDAVecRestoreArray(fda, Centz, &centz);
  DMDAVecRestoreArray(fda, Centy, &centy);
  DMDAVecRestoreArray(fda, Centx, &centx);
  
	DMDALocalToLocalBegin(fda, Centx, INSERT_VALUES, Centx);
	DMDALocalToLocalEnd(fda, Centx, INSERT_VALUES, Centx);
	
	DMDALocalToLocalBegin(fda, Centy, INSERT_VALUES, Centy);
	DMDALocalToLocalEnd(fda, Centy, INSERT_VALUES, Centy);
	
	DMDALocalToLocalBegin(fda, Centz, INSERT_VALUES, Centz);
	DMDALocalToLocalEnd(fda, Centz, INSERT_VALUES, Centz);
	
/*
  DMDAVecRestoreArray(cda, Csi, &csi);
  DMDAVecRestoreArray(cda, Eta, &eta);
  DMDAVecRestoreArray(cda, Zet, &zet);
  DMDAVecRestoreArray(da, Aj,  &aj);
*/
  DMDAVecRestoreArray(fda, KCsi, &kcsi);
  DMDAVecRestoreArray(fda, KEta, &keta);
  DMDAVecRestoreArray(fda, KZet, &kzet);
  DMDAVecRestoreArray(da, KAj,  &kaj);
  

  DMDAVecRestoreArray(cda, coords, &coor);


  VecAssemblyBegin(Csi);
  VecAssemblyEnd(Csi);
  VecAssemblyBegin(Eta);
  VecAssemblyEnd(Eta);
  VecAssemblyBegin(Zet);
  VecAssemblyEnd(Zet);
  VecAssemblyBegin(Aj);
  VecAssemblyEnd(Aj);

  VecAssemblyBegin(user->Cent);
  VecAssemblyEnd(user->Cent);

  VecAssemblyBegin(user->ICsi);
  VecAssemblyEnd(user->ICsi);
  VecAssemblyBegin(user->IEta);
  VecAssemblyEnd(user->IEta);
  VecAssemblyBegin(user->IZet);
  VecAssemblyEnd(user->IZet);
  VecAssemblyBegin(user->IAj);
  VecAssemblyEnd(user->IAj);

  VecAssemblyBegin(user->JCsi);
  VecAssemblyEnd(user->JCsi);
  VecAssemblyBegin(user->JEta);
  VecAssemblyEnd(user->JEta);
  VecAssemblyBegin(user->JZet);
  VecAssemblyEnd(user->JZet);
  VecAssemblyBegin(user->JAj);
  VecAssemblyEnd(user->JAj);

  VecAssemblyBegin(user->KCsi);
  VecAssemblyEnd(user->KCsi);
  VecAssemblyBegin(user->KEta);
  VecAssemblyEnd(user->KEta);
  VecAssemblyBegin(user->KZet);
  VecAssemblyEnd(user->KZet);
  VecAssemblyBegin(user->KAj);
  VecAssemblyEnd(user->KAj);

  DMRestoreLocalVector(fda, &Centx);

  VecDestroy(&Centy);
  VecDestroy(&Centz);

  DMGlobalToLocalBegin(fda, user->Csi, INSERT_VALUES, user->lCsi);
  DMGlobalToLocalEnd(fda, user->Csi, INSERT_VALUES, user->lCsi);

  DMGlobalToLocalBegin(fda, user->Eta, INSERT_VALUES, user->lEta);
  DMGlobalToLocalEnd(fda, user->Eta, INSERT_VALUES, user->lEta);

  DMGlobalToLocalBegin(fda, user->Zet, INSERT_VALUES, user->lZet);
  DMGlobalToLocalEnd(fda, user->Zet, INSERT_VALUES, user->lZet);

  DMGlobalToLocalBegin(fda, user->ICsi, INSERT_VALUES, user->lICsi);
  DMGlobalToLocalEnd(fda, user->ICsi, INSERT_VALUES, user->lICsi);

  DMGlobalToLocalBegin(fda, user->IEta, INSERT_VALUES, user->lIEta);
  DMGlobalToLocalEnd(fda, user->IEta, INSERT_VALUES, user->lIEta);

  DMGlobalToLocalBegin(fda, user->IZet, INSERT_VALUES, user->lIZet);
  DMGlobalToLocalEnd(fda, user->IZet, INSERT_VALUES, user->lIZet);

  DMGlobalToLocalBegin(fda, user->JCsi, INSERT_VALUES, user->lJCsi);
  DMGlobalToLocalEnd(fda, user->JCsi, INSERT_VALUES, user->lJCsi);

  DMGlobalToLocalBegin(fda, user->JEta, INSERT_VALUES, user->lJEta);
  DMGlobalToLocalEnd(fda, user->JEta, INSERT_VALUES, user->lJEta);

  DMGlobalToLocalBegin(fda, user->JZet, INSERT_VALUES, user->lJZet);
  DMGlobalToLocalEnd(fda, user->JZet, INSERT_VALUES, user->lJZet);

  DMGlobalToLocalBegin(fda, user->KCsi, INSERT_VALUES, user->lKCsi);
  DMGlobalToLocalEnd(fda, user->KCsi, INSERT_VALUES, user->lKCsi);

  DMGlobalToLocalBegin(fda, user->KEta, INSERT_VALUES, user->lKEta);
  DMGlobalToLocalEnd(fda, user->KEta, INSERT_VALUES, user->lKEta);

  DMGlobalToLocalBegin(fda, user->KZet, INSERT_VALUES, user->lKZet);
  DMGlobalToLocalEnd(fda, user->KZet, INSERT_VALUES, user->lKZet);

  DMGlobalToLocalBegin(da, user->Aj, INSERT_VALUES, user->lAj);
  DMGlobalToLocalEnd(da, user->Aj, INSERT_VALUES, user->lAj);

  DMGlobalToLocalBegin(da, user->IAj, INSERT_VALUES, user->lIAj);
  DMGlobalToLocalEnd(da, user->IAj, INSERT_VALUES, user->lIAj);

  DMGlobalToLocalBegin(da, user->JAj, INSERT_VALUES, user->lJAj);
  DMGlobalToLocalEnd(da, user->JAj, INSERT_VALUES, user->lJAj);

  DMGlobalToLocalBegin(da, user->KAj, INSERT_VALUES, user->lKAj);
  DMGlobalToLocalEnd(da, user->KAj, INSERT_VALUES, user->lKAj);

  DMGlobalToLocalBegin(fda, user->GridSpace, INSERT_VALUES, user->lGridSpace);
  DMGlobalToLocalEnd(fda, user->GridSpace, INSERT_VALUES, user->lGridSpace);

  DMGlobalToLocalBegin(fda, user->Cent, INSERT_VALUES, user->lCent);
  DMGlobalToLocalEnd(fda, user->Cent, INSERT_VALUES, user->lCent);

  VecDestroy(&user->Csi);
  VecDestroy(&user->Eta);
  VecDestroy(&user->Zet);

  VecDestroy(&user->ICsi);
  VecDestroy(&user->IEta);
  VecDestroy(&user->IZet);

  VecDestroy(&user->JCsi);
  VecDestroy(&user->JEta);
  VecDestroy(&user->JZet);

  VecDestroy(&user->KCsi);
  VecDestroy(&user->KEta);
  VecDestroy(&user->KZet);

  VecDestroy(&user->Aj);
  VecDestroy(&user->IAj);
  VecDestroy(&user->JAj);
  VecDestroy(&user->KAj);

  PetscBarrier(PETSC_NULL);

return 0;
}

