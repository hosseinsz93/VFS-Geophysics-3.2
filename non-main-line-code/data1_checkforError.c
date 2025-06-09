static char help[] = "Testing programming!";

#include <vector>
#include "petscda.h"
#include "petscts.h"
#include "petscpc.h"
#include "petscsnes.h"
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
using namespace std;


#define NEWMETRIC

#ifdef TECIO  
#include "TECIO.h"
#endif 
PetscInt ti, block_number, Flux_in;
int binary_input=0;
int xyz_input=0;
PetscInt tis, tie, tsteps=5, steps_avg;
PetscReal angle;
int nv_once=0;
int onlyV=0;
int k_average=0;
int j_average=0;
int i_average=0;
int ik_average=0;
int ikc_average=0;	// conditional spatial averaging in ik directions (channel flow)
int reynolds=0;	// 1: contravariant reynolds stress

int i_begin, i_end;
int j_begin, j_end;
int k_begin, k_end;

int pcr=0;
int avg=0, rans=0, rans_output=0, levelset=0;
int vc = 1;

int cs=0;
int i_periodic=0;
int j_periodic=0;
int k_periodic=0;
int kk_periodic=0;
int averaging_option=0;
int pi=-1, pk=-1;
int shear=0;

char path[256];
int LES=0;

int les_prt, temperature;

char prefix[256];

int rotor_model = 0;

int NumberOfTurbines = 0;

int deltafunc; 

PetscReal L_dim = 1.0;

PetscReal CMx_c = 0.0, CMy_c = 0.0, CMz_c = 0.0;
//int l, m, n;
/* Symmetric matrix A -> eigenvectors in columns of V, corresponding eigenvalues in d. */
void eigen_decomposition(double A[3][3], double V[3][3], double d[3]);


typedef struct {
  PetscReal t, f;
} FlowWave;

typedef struct {
  PassiveScalar u, v, w, p;
} PassiveField;

typedef struct {
  PetscScalar u, v, w;
} Field;

typedef struct {
  PetscScalar x, y, z;
} Cmpnts;

typedef struct {
  PetscScalar x, y;
} Cmpnts2;

typedef struct {
  PassiveScalar csi[3], eta[3], zet[3], aj;
} Metrics;

typedef struct {
  Vec	Ubcs; // An ugly hack, waste of memory
} BCS;



typedef struct {
	PetscInt	nbnumber;
	PetscInt	n_v, n_elmt;	// number of vertices and number of elements
	PetscInt	my_n_v, my_n_elmt;	// seokkoo, my proc
	PetscInt	*nv1, *nv2, *nv3;	// Node index of each triangle
	PetscReal	*nf_x, *nf_y, *nf_z;	// Normal direction
	//PetscReal	*nf_x0, *nf_y0, *nf_z0;	// Normal direction
	PetscReal	*x_bp, *y_bp, *z_bp;	// Coordinates of IBM surface nodes
	PetscReal	*x_bp0, *y_bp0, *z_bp0;
	PetscReal	*x_bp_o, *y_bp_o, *z_bp_o;
	//PetscReal	x_bp_in[101][3270], y_bp_in[101][3270], z_bp_in[101][3270];
	Cmpnts	*u, *uold, *urm1;
  
	// Added 4/1/06 iman
	PetscReal     *dA ;         // area of an element
	// Added 4/2/06 iman
	PetscReal     *nt_x, *nt_y, *nt_z; //tangent dir
	PetscReal     *ns_x, *ns_y, *ns_z; //azimuthal dir
	// Added 6/4/06 iman
	//Cmpnts        *cent; // center of the element 
	PetscReal     *cent_x,*cent_y,*cent_z;

	// for radius check
	Cmpnts *qvec;
	PetscReal *radvec; 
	
	//seokkoo
	
	PetscInt *_nv1, *_nv2, *_nv3;	// for rank 0
	PetscInt *count;
	PetscInt *local2global_elmt;
	int total_n_elmt, total_n_v;
	
	PetscReal *_x_bp, *_y_bp, *_z_bp;	// for rank 0
	PetscReal *shear, *mean_shear;
	PetscReal *reynolds_stress1;
	PetscReal *reynolds_stress2;
	PetscReal *reynolds_stress3;
	PetscReal *pressure;

	// xyang
	/* for calculating surface force */
	PetscInt	*ib_elmt, *jb_elmt, *kb_elmt; // the lowest interpolation point for element

	PetscReal	*xb_elmt, *yb_elmt, *zb_elmt; // normal extension from the surface element center

	PetscReal	*p_elmt;

	Cmpnts		*tau_elmt;


        // xyang begin 12-7-2010
        PetscReal 	*F_lagr_x, *F_lagr_y, *F_lagr_z; // force at the IB surface points (lagrange points)
        PetscReal 	*U_lagr_x, *U_lagr_y, *U_lagr_z;
        PetscInt        *i_min, *i_max, *j_min, *j_max, *k_min, *k_max;

        /* ACL */
        PetscReal       *angle_attack, *angle_pitch, *chord_blade; // twist angle and chord length at each grid point
        PetscReal       *CD, *CL;

	PetscReal 	U_ref;
        // xyang end

	// 
	PetscReal	*dh_g, *dh_e; // The length extended in the normal direction
        PetscReal     	*x_bp_g, *y_bp_g, *z_bp_g;    // Coordinates of IBM surface ghost nodes
        PetscReal     	*dA_g;         // area of an element of the ghost points
        PetscReal     	*cent_x_g,*cent_y_g,*cent_z_g;

        PetscReal       *x_bp_e, *y_bp_e, *z_bp_e;    // Coordinates of IBM surface exterior nodes
        PetscReal       *dA_e;         // area of an element of the exterior points
        PetscReal       *cent_x_e,*cent_y_e,*cent_z_e;

        PetscInt        *i_min_e, *i_max_e, *j_min_e, *j_max_e, *k_min_e, *k_max_e;
        PetscInt        *i_min_g, *i_max_g, *j_min_g, *j_max_g, *k_min_g, *k_max_g;

	PetscReal	*dhx_g, *dhy_g, *dhz_g, *dhx_e, *dhy_e, *dhz_e; // The background grid width near the immersed boundary

	PetscReal	*dhx, *dhy, *dhz;

        PetscReal 	*U_lagrx_g, *U_lagry_g, *U_lagrz_g; // Lagrangian velocity at the ghost points

        PetscReal 	*U_lagrx_e, *U_lagry_e, *U_lagrz_e; // Lagrangian velocity at the ghost points

        // xyang end
	
} IBMNodes;


typedef struct {
  PetscInt	IM, JM, KM; // dimensions of grid
  DA da;	/* Data structure for scalars (include the grid geometry
		   informaion, to obtain the grid information, 
		   use DAGetCoordinates) */
  DA fda, fda2;	// Data Structure for vectors
  DALocalInfo info;

  Vec	Cent;	// Coordinates of cell centers
  Vec 	Csi, Eta, Zet, Aj;
  Vec 	ICsi, IEta, IZet, IAj;
  Vec 	JCsi, JEta, JZet, JAj;
  Vec 	KCsi, KEta, KZet, KAj;
  Vec 	Ucont;	// Contravariant velocity components
  Vec 	Ucat;	// Cartesian velocity components
  Vec	Ucat_o;
  Vec 	P;
  Vec	Phi;
  Vec	GridSpace;
  Vec	Nvert;
  Vec	Nvert_o;
  BCS	Bcs;

  PetscInt	*nvert;//ody property

  PetscReal	ren;	// Reynolds number
  PetscReal	dt; 	// time step
  PetscReal	st;	// Strouhal number

  PetscReal	r[101], tin[101], uinr[101][1001];

  Vec	lUcont, lUcat, lP, lPhi;
  Vec	lCsi, lEta, lZet, lAj;
  Vec	lICsi, lIEta, lIZet, lIAj;
  Vec	lJCsi, lJEta, lJZet, lJAj;
  Vec	lKCsi, lKEta, lKZet, lKAj;
  Vec	lGridSpace;
  Vec	lNvert, lNvert_o;
  Vec	lCent;
  
  Vec Ucat_sum;		// u, v, w
  Vec Ucat_cross_sum;		// uv, vw, wu
  Vec Ucat_square_sum;	// u^2, v^2, w^2

  Vec Ustar;   

  PetscInt _this;

  FlowWave *inflow, *kinematics;
  PetscInt number_flowwave, number_kinematics;

} UserCtx;

typedef struct {
  PetscInt	i1, j1, k1;
  PetscInt	i2, j2, k2;
  PetscInt	i3, j3, k3;
  PetscReal	cr1, cr2, cr3; // coefficients
  PetscReal	d_i; // distance to interception point on grid cells
  PetscInt	imode; // interception mode

  PetscInt	ni, nj, nk;	// fluid cell
  PetscReal	d_s; // shortest distance to solid surfaces
  Cmpnts	pmin;
  PetscInt	cell; // shortest distance surface cell
  PetscReal	cs1, cs2, cs3;

  PetscInt	i11, j11, k11;
  PetscInt	i22, j22, k22;
  PetscInt	i33, j33, k33;
  PetscReal	cr11, cr22, cr33; // coefficients
  PetscReal	d_ii; // distance to interception point on grid cells
  PetscInt	iimode; // interception mode
  PetscReal	cs11, cs22, cs33;

  PetscInt	ii1, jj1, kk1;
  PetscInt	ii2, jj2, kk2;
  PetscInt	ii3, jj3, kk3;
  PetscReal	ct1, ct2, ct3; // coefficients
  //PetscReal	d_s; // distance to interception point on grid cells
  PetscInt	smode; // interception mode

  PetscInt	ii11, jj11, kk11;
  PetscInt	ii22, jj22, kk22;
  PetscInt	ii33, jj33, kk33;
  PetscReal	ct11, ct22, ct33; // coefficients
  PetscReal	d_ss; // distance to interception point on grid cells
  PetscInt	ssmode; // interception mode

  
/*   PetscInt      bi1, bj1, bk1; */
/*   PetscInt      bi2, bj2, bk2; */
/*   PetscInt      bi3, bj3, bk3; */
/*   PetscInt      bi4, bj4, bk4; */

/*   PetscReal     bcr1,bcr2,bcr3,bcr4; */
} IBMInfo;

typedef struct {
        PetscReal       P;    //Press on the surface elmt
        PetscInt                n_P; //number of Press Pts on the elmt
        PetscReal       Tow_ws, Tow_wt; //wall shear stress of the elmt
        PetscReal       Tow_wn; // normal stress

        PetscInt                Clsnbpt_i,Clsnbpt_j,Clsnbpt_k; //Closest Near Bndry Pt to each surf elmt
        PetscInt                icell,jcell,kcell;
        PetscInt                FoundAroundcell;
        PetscInt                Need3rdPoint;
        //PetscInt      Aroundcellnum;
} SurfElmtInfo;

typedef struct {
	PetscReal    S_new[6],S_old[6],S_real[6],S_realm1[6];
	PetscReal    S_ang_n[6],S_ang_o[6],S_ang_r[6],S_ang_rm1[6];
	PetscReal    red_vel, damp, mu_s; // reduced vel, damping coeff, mass coeff
	PetscReal    F_x,F_y,F_z, A_tot; //Forces & Area
	PetscReal    F_x_old,F_y_old,F_z_old; //Forces & Area
	PetscReal    F_x_real,F_y_real,F_z_real; //Forces & Area
	PetscReal    M_x,M_y,M_z; // Moments
	PetscReal    M_x_old,M_y_old,M_z_old; //Forces & Area
	PetscReal    M_x_real,M_y_real,M_z_real; //Forces & Area
	PetscReal    M_x_rm2,M_y_rm2,M_z_rm2; //Forces & Area
	PetscReal    M_x_rm3,M_y_rm3,M_z_rm3; //Forces & Area
	PetscReal    x_c,y_c,z_c; // center of rotation(mass)
	PetscReal    Mdpdn_x, Mdpdn_y,Mdpdn_z;
	PetscReal    Mdpdn_x_old, Mdpdn_y_old,Mdpdn_z_old;

	PetscReal    angvel_z, angvel_x, angvel_y;


	// Aitkin's iteration
	PetscReal    dS[6],dS_o[6],atk,atk_o;

	// for force calculation
	SurfElmtInfo  *elmtinfo;
	IBMInfo       *fsi_intp;

	//Max & Min of ibm domain where forces are calculated
	PetscReal    Max_xbc,Min_xbc;
	PetscReal    Max_ybc,Min_ybc;
	PetscReal    Max_zbc,Min_zbc;

	// CV bndry
	PetscInt     CV_ys,CV_ye,CV_zs,CV_ze;

	PetscReal    omega_x, omega_y, omega_z; //xyang
		
}	FSInfo;


PetscErrorCode ReadCoordinates(UserCtx *user);
PetscErrorCode QCriteria(UserCtx *user);
PetscErrorCode Velocity_Magnitude(UserCtx *user);
PetscErrorCode Lambda2(UserCtx *user);
PetscErrorCode FormMetrics(UserCtx *user);
void Calc_avg_shear_stress(UserCtx *user);
PetscErrorCode ACD_read(IBMNodes *ibm, PetscInt ibi, FSInfo *fsi);

PetscErrorCode Pre_process(UserCtx *user, IBMNodes *ibm);

PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, Vec v_eul, char fname[80]);

double dfunc_4h(double r);

double dfunc_2h(double r);
double dfunc_s3h(double r);


int file_exist(char *str)
{
  int r=0;

  /*if(!my_rank)*/ {
    FILE *fp=fopen(str, "r");
    if(!fp) {
      r=0;
      printf("\nFILE !!! %s does not exist !!!\n", str);
    }
    else {
      fclose(fp);
      r=1;
    }
  }
  MPI_Bcast(&r, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  return r;
};

void Calculate_Covariant_metrics(double g[3][3], double G[3][3])
{
	/*
		| csi.x  csi.y csi.z |-1		| x.csi  x.eta x.zet | 
		| eta.x eta.y eta.z |	 =	| y.csi   y.eta  y.zet |
		| zet.x zet.y zet.z |		| z.csi  z.eta z.zet |
	
	*/
	const double a11=g[0][0], a12=g[0][1], a13=g[0][2];
	const double a21=g[1][0], a22=g[1][1], a23=g[1][2];
	const double a31=g[2][0], a32=g[2][1], a33=g[2][2];

	double det= a11*(a33*a22-a32*a23)-a21*(a33*a12-a32*a13)+a31*(a23*a12-a22*a13);
	
	G[0][0] = (a33*a22-a32*a23)/det,	G[0][1] = - (a33*a12-a32*a13)/det, 	G[0][2] = (a23*a12-a22*a13)/det;
	G[1][0] = -(a33*a21-a31*a23)/det, G[1][1] = (a33*a11-a31*a13)/det,	G[1][2] = - (a23*a11-a21*a13)/det;
	G[2][0] = (a32*a21-a31*a22)/det,	G[2][1] = - (a32*a11-a31*a12)/det,	G[2][2] = (a22*a11-a21*a12)/det;
};

void Calculate_normal(Cmpnts csi, Cmpnts eta, Cmpnts zet, double ni[3], double nj[3], double nk[3])
{
	double g[3][3];
	double G[3][3];
	
	g[0][0]=csi.x, g[0][1]=csi.y, g[0][2]=csi.z;
	g[1][0]=eta.x, g[1][1]=eta.y, g[1][2]=eta.z;
	g[2][0]=zet.x, g[2][1]=zet.y, g[2][2]=zet.z;
	
	Calculate_Covariant_metrics(g, G);
	double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
	double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
	double xzet=G[0][2], yzet=G[1][2], zzet=G[2][2];
	      
	double nx_i = xcsi, ny_i = ycsi, nz_i = zcsi;
	double nx_j = xeta, ny_j = yeta, nz_j = zeta;
	double nx_k = xzet, ny_k = yzet, nz_k = zzet;
	      
	double sum_i=sqrt(nx_i*nx_i+ny_i*ny_i+nz_i*nz_i);
	double sum_j=sqrt(nx_j*nx_j+ny_j*ny_j+nz_j*nz_j);
	double sum_k=sqrt(nx_k*nx_k+ny_k*ny_k+nz_k*nz_k);

	nx_i /= sum_i, ny_i /= sum_i, nz_i /= sum_i;
	nx_j /= sum_j, ny_j /= sum_j, nz_j /= sum_j;
	nx_k /= sum_k, ny_k /= sum_k, nz_k /= sum_k;

	ni[0] = nx_i, ni[1] = ny_i, ni[2] = nz_i;
	nj[0] = nx_j, nj[1] = ny_j, nj[2] = nz_j;
	nk[0] = nx_k, nk[1] = ny_k, nk[2] = nz_k;
}

double Contravariant_Reynolds_stress(double uu, double uv, double uw, double vv, double vw, double ww,
	double csi0, double csi1, double csi2, double eta0, double eta1, double eta2)
{
	
	double A = uu*csi0*eta0 + vv*csi1*eta1 + ww*csi2*eta2 + uv * (csi0*eta1+csi1*eta0)	+ uw * (csi0*eta2+csi2*eta0) + vw * (csi1*eta2+csi2*eta1);
	double B = sqrt(csi0*csi0+csi1*csi1+csi2*csi2)*sqrt(eta0*eta0+eta1*eta1+eta2*eta2);
	
	return A/B;
}

void Compute_du_center (int i, int j, int k,  int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert, 
				double *dudc, double *dvdc, double *dwdc, 
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	if ((nvert[k][j][i+1])> 0.1) {
		*dudc = ( ucat[k][j][i].x - ucat[k][j][i-1].x );
		*dvdc = ( ucat[k][j][i].y - ucat[k][j][i-1].y );
		*dwdc = ( ucat[k][j][i].z - ucat[k][j][i-1].z );
	}
	else if ((nvert[k][j][i-1])> 0.1) {
		*dudc = ( ucat[k][j][i+1].x - ucat[k][j][i].x );
		*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i].y );
		*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i].z );
	}
	else {
		if(i_periodic && i==1) {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][mx-2].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][mx-2].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][mx-2].z ) * 0.5;
		}
		else if(i_periodic && i==mx-2) {
			*dudc = ( ucat[k][j][1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][1].z - ucat[k][j][i-1].z ) * 0.5;
		}
		else {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i-1].z ) * 0.5;
		}
	}

	if ((nvert[k][j+1][i])> 0.1) {
		*dude = ( ucat[k][j][i].x - ucat[k][j-1][i].x );
		*dvde = ( ucat[k][j][i].y - ucat[k][j-1][i].y );
		*dwde = ( ucat[k][j][i].z - ucat[k][j-1][i].z );
	}
	else if ((nvert[k][j-1][i])> 0.1) {
		*dude = ( ucat[k][j+1][i].x - ucat[k][j][i].x );
		*dvde = ( ucat[k][j+1][i].y - ucat[k][j][i].y );
		*dwde = ( ucat[k][j+1][i].z - ucat[k][j][i].z );
	}
	else {
		if(j_periodic && j==1) {
			*dude = ( ucat[k][j+1][i].x - ucat[k][my-2][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][my-2][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][my-2][i].z ) * 0.5;
		}
		else if(j_periodic && j==my-2) {
			*dude = ( ucat[k][1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
		else {
			*dude = ( ucat[k][j+1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
	}

	if ((nvert[k+1][j][i])> 0.1) {
		*dudz = ( ucat[k][j][i].x - ucat[k-1][j][i].x );
		*dvdz = ( ucat[k][j][i].y - ucat[k-1][j][i].y );
		*dwdz = ( ucat[k][j][i].z - ucat[k-1][j][i].z );
	}
	else if ((nvert[k-1][j][i])> 0.1) {
		*dudz = ( ucat[k+1][j][i].x - ucat[k][j][i].x );
		*dvdz = ( ucat[k+1][j][i].y - ucat[k][j][i].y );
		*dwdz = ( ucat[k+1][j][i].z - ucat[k][j][i].z );
	}
	else {
		if(k_periodic && k==1) {
			*dudz = ( ucat[k+1][j][i].x - ucat[mz-2][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[mz-2][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[mz-2][j][i].z ) * 0.5;
		}
		else if(k_periodic && k==mz-2) {
			*dudz = ( ucat[1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
		else {
			*dudz = ( ucat[k+1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
	}
}


void Compute_du_dxyz (	double csi0, double csi1, double csi2, double eta0, double eta1, double eta2, double zet0, double zet1, double zet2, double ajc,
					double dudc, double dvdc, double dwdc, double dude, double dvde, double dwde, double dudz, double dvdz, double dwdz,
					double *du_dx, double *dv_dx, double *dw_dx, double *du_dy, double *dv_dy, double *dw_dy, double *du_dz, double *dv_dz, double *dw_dz )
{
	*du_dx = (dudc * csi0 + dude * eta0 + dudz * zet0) * ajc;
	*du_dy = (dudc * csi1 + dude * eta1 + dudz * zet1) * ajc;
	*du_dz = (dudc * csi2 + dude * eta2 + dudz * zet2) * ajc;
	*dv_dx = (dvdc * csi0 + dvde * eta0 + dvdz * zet0) * ajc;
	*dv_dy = (dvdc * csi1 + dvde * eta1 + dvdz * zet1) * ajc;
	*dv_dz = (dvdc * csi2 + dvde * eta2 + dvdz * zet2) * ajc;
	*dw_dx = (dwdc * csi0 + dwde * eta0 + dwdz * zet0) * ajc;
	*dw_dy = (dwdc * csi1 + dwde * eta1 + dwdz * zet1) * ajc;	
	*dw_dz = (dwdc * csi2 + dwde * eta2 + dwdz * zet2) * ajc;
};


void IKavg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (j=ys; j<ye-2; j++) {
		double iksum=0;
		int count=0;
		for (i=xs; i<xe-2; i++)
		for (k=zs; k<ze-2; k++) {
			iksum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
			count++;
		}
		double ikavg = iksum/(double)count;
		for (i=xs; i<xe-2; i++)
		for (k=zs; k<ze-2; k++) 	x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ikavg;
	}
}

/*
	pi, pk : # of grid points correcsponding to the period
	conditional averaging
*/
void IKavg_c (float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
  //int i, j, k;
	
  if(pi<=0) pi = (xe-xs-2); // no averaging
  if(pk<=0) pk = (ze-zs-2); // no averaging 
	
	int ni = (xe-xs-2) / pi;
	int nk = (ze-zs-2) / pk;

	std::vector< std::vector<float> > iksum (pk);
	
	for(int k=0; k<pk; k++) iksum[k].resize(pi);
	
	for (int j=ys; j<ye-2; j++) {
		
		for(int k=0; k<pk; k++) std::fill( iksum[k].begin(), iksum[k].end(), 0.0 );
		
		for (int i=xs; i<xe-2; i++)
		for (int k=zs; k<ze-2; k++) {
			iksum [ ( k - zs ) % pk ] [ ( i - xs ) % pi] += x [k * (mx-2)*(my-2) + j*(mx-2) + i];
		}

		for (int i=xs; i<xe-2; i++)
		for (int k=zs; k<ze-2; k++) x [k * (mx-2)*(my-2) + j*(mx-2) + i] = iksum [ ( k - zs ) % pk ] [ ( i - xs ) % pi ] / (ni*nk);
	}
}


void Kavg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (j=ys; j<ye-2; j++) 
	for (i=xs; i<xe-2; i++) {
	  double ksum=0;
	  int count=0;
	  for (k=zs; k<ze-2; k++) {
		ksum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
		count++;
	  }
	  double kavg = ksum/(double)count;
	  for (k=zs; k<ze-2; k++) {
		x[k * (mx-2)*(my-2) + j*(mx-2) + i] = kavg;
	  }
	}
      
}

void Javg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;
	
	for (k=zs; k<ze-2; k++)
	for (i=xs; i<xe-2; i++) {
	  double jsum=0;
	  int count=0;
	  for (j=ys; j<ye-2; j++) {
		jsum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
		count++;
	  }
	  double javg = jsum/(double)count;
	  for (j=ys; j<ye-2; j++) {
		x[k * (mx-2)*(my-2) + j*(mx-2) + i] = javg;
	  }
	}
      
}

void Iavg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;
	
	for (k=zs; k<ze-2; k++) {
	  for (j=ys; j<ye-2; j++) {
	    double isum=0;
	    int count=0;
	    for (i=xs; i<xe-2; i++) {
	      isum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
	      count++;
	    }
	    double iavg = isum/(double)count;
	    for (i=xs; i<xe-2; i++) {
	      x[k * (mx-2)*(my-2) + j*(mx-2) + i] = iavg;
	    }
	  }
	}
}

void Iavg(Cmpnts ***x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;
	
	for (k=zs; k<ze-2; k++) 
	for (j=ys; j<ye-2; j++) {
		Cmpnts isum, iavg;
		isum.x = isum.y = isum.z = 0;
	    
	  int count=0;
		for (i=xs; i<xe-2; i++) {
	     isum.x += x[k+1][j+1][i+1].x;
	     isum.y += x[k+1][j+1][i+1].y;
	     isum.z += x[k+1][j+1][i+1].z;
	     count++;
	  }
	  
	  iavg.x = isum.x /(double)count;
	  iavg.y = isum.y /(double)count;
	  iavg.z = isum.z /(double)count;
	    
		for (i=xs; i<xe-2; i++) {
	  	x[k+1][j+1][i+1] = iavg;
		}
	}
}

void Iavg(PetscReal ***x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;
	
	for (k=zs; k<ze-2; k++) 
	for (j=ys; j<ye-2; j++) {
		double isum, iavg;
		isum = 0;
	    
	  int count=0;
		for (i=xs; i<xe-2; i++) {
	     isum += x[k+1][j+1][i+1];
	     count++;
	  }
	  iavg = isum /(double)count;
	    
		for (i=xs; i<xe-2; i++) {
	  	x[k+1][j+1][i+1] = iavg;
		}
	}
}

PetscErrorCode TECIOOut_V_2D(UserCtx *user, int only_V)	// seokkoo
{
	PetscInt bi;
	PetscInt IMax, JMax, KMax;
        PetscReal x, y, z, u, v, w, p1;
	char filen[80];

	double xy_plane = 0.1;
	double yz_plane = 0.1;
	double zx_plane = 0.1;

 	PetscOptionsGetReal(PETSC_NULL, "-xy_plane", &(xy_plane), PETSC_NULL);
 	PetscOptionsGetReal(PETSC_NULL, "-yz_plane", &(yz_plane), PETSC_NULL);
 	PetscOptionsGetReal(PETSC_NULL, "-zx_plane", &(zx_plane), PETSC_NULL);


	PetscInt ix_yz, iy_zx, iz_xy;


//	FILE * pFile1;	
//	sprintf(filen, "%sResult-Vorticity%06d.plt", prefix, ti);
//        pFile1 = fopen(filen, "w");

        sprintf(filen, "%sResult-xy2d%g-%06d.plt", prefix, xy_plane, ti);
        FILE * pFile;
        pFile = fopen(filen, "w");
        fprintf(pFile, " TITLE = \"Flow Field\" \n");
        fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"W\", \"P\", \"nv\" \n" );

//        fprintf(pFile1, " TITLE = \"Vorticity Field\" \n");
//        fprintf(pFile1, " VARIABLES = \"X\", \"Y\", \"Z\", \"vor-1\", \"vor-2\", \"vor-3\", \"nv\" \n" );

//        PetscReal       ***aj;
//        Cmpnts  ***csi, ***eta, ***zet;

//  	DAVecGetArray(user->fda, user->Csi, &csi);
//  	DAVecGetArray(user->fda, user->Eta, &eta);
//  	DAVecGetArray(user->fda, user->Zet, &zet);
//  	DAVecGetArray(user->da, user->Aj, &aj);

        sprintf(filen, "%sResult-yz2d%g-%06d.plt", prefix, yz_plane, ti);
        FILE * pFile1;
        pFile1 = fopen(filen, "w");
        fprintf(pFile1, " TITLE = \"Flow Field\" \n");
        fprintf(pFile1, " VARIABLES = \"Z\", \"Y\", \"U\", \"V\", \"W\", \"P\", \"nv\" \n" );


        sprintf(filen, "%sResult-zx2d%g-%06d.plt", prefix, zx_plane, ti);
        FILE * pFile2;
        pFile2 = fopen(filen, "w");
        fprintf(pFile2, " TITLE = \"Flow Field\" \n");
        fprintf(pFile2, " VARIABLES = \"Z\", \"X\", \"U\", \"V\", \"W\", \"P\", \"nv\" \n" );



	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;
    
		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o, ***csi, ***eta, ***zet;
		PetscReal	***p, ***nvert, ***level;
		Vec		Coor, zCoor, nCoor;
		VecScatter	ctx;
		Vec K_Omega;
		PetscReal	uu;	
		PetscReal	vor_1, vor_2, vor_3;	

		DAVecGetArray(user[bi].da, user[bi].Nvert, &nvert);
		DAVecGetArray(user[bi].fda, user[bi].Ucat, &ucat);
		DAVecGetArray(user[bi].da, user[bi].P, &p);


//  		DAVecGetArray(user[bi].fda, user[bi].Csi, &csi);
//  		DAVecGetArray(user[bi].fda, user[bi].Eta, &eta);
//  		DAVecGetArray(user[bi].fda, user[bi].Zet, &zet);
//  		DAVecGetArray(user[bi].da, user[bi].Aj, &aj);

		
		IMax = mx-1;
		JMax = my-1;
		KMax = mz-1;

		fprintf(pFile, "ZONE I=%d, J=%d, DATAPACKING=POINT \n", IMax, JMax );

                fprintf(pFile1, "ZONE I=%d, J=%d, DATAPACKING=POINT \n", KMax, JMax );

                fprintf(pFile2, "ZONE I=%d, J=%d, DATAPACKING=POINT \n", KMax, IMax );

		
//		fprintf(pFile1, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax-1, JMax-1, KMax-1 );

		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);
        
		j = 1;
		k = 1;
		for (i=xs; i<xe-2; i++){
			if (coor[k][j][i].x <= yz_plane && coor[k][j][i+1].x > yz_plane){
				ix_yz = i; 
				printf("the x dir %d %le %le %le \n", ix_yz, coor[k][j][i].x, yz_plane,  coor[k][j][i+1].x);
			}
		}

                i = 1;
                k = 1;
                for (j=ys; j<ye-2; j++){
                        if (coor[k][j][i].y <= zx_plane && coor[k][j+1][i].y > zx_plane) {
				iy_zx = j;
				printf("the y dir %d %le %le %le \n", iy_zx, coor[k][j][i].y, zx_plane,  coor[k][j+1][i].y);
			}
                }

                i = 1;
                j = 1;
                for (k=zs; k<ze-2; k++){
                        if (coor[k][j][i].z <= xy_plane && coor[k+1][j][i].z > xy_plane) {
				iz_xy = k;

				printf("the z dir %d %le %le %le \n", iz_xy, coor[k][j][i].z, xy_plane,  coor[k+1][j][i].z);
			}
                }

		printf("the ix, iy, iz %d %d %d \n", ix_yz, iy_zx, iz_xy);

		double fac1, fac2;

		k = iz_xy;
                for (j=ys; j<ye-1; j++){
                for (i=xs; i<xe-1; i++){
			fac1 = (coor[k+1][j][i].z - xy_plane)/(coor[k+1][j][i].z - coor[k][j][i].z);
			fac2 = (-coor[k][j][i].z + xy_plane)/(coor[k+1][j][i].z - coor[k][j][i].z);

                        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        u = ucat[k][j][i].x * fac1 + ucat[k+1][j][i].x * fac2;
                        v = ucat[k][j][i].y * fac1 + ucat[k+1][j][i].y * fac2;
                        w = ucat[k][j][i].z * fac1 + ucat[k+1][j][i].z * fac2;
                        p1 = p[k][j][i] * fac1 + p[k+1][j][i] * fac2;
                        uu = sqrt( u*u + v*v + w*w );
                        fprintf(pFile, "%le %le %le %le %le %le %le \n", x, y, 
                                u, v, w, p1, nvert[k][j][i] );

                }
                }

                fclose(pFile);

		//printf(" HERE 1 \n");


                i = ix_yz;
                for (j=ys; j<ye-1; j++){
                for (k=zs; k<ze-1; k++){
                        fac1 = (coor[k][j][i+1].x - yz_plane)/(coor[k][j][i+1].x - coor[k][j][i].x);
                        fac2 = (-coor[k][j][i].x + yz_plane)/(coor[k][j][i+1].x - coor[k][j][i].x);

                        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        u = ucat[k][j][i].x * fac1 + ucat[k][j][i+1].x * fac2;
                        v = ucat[k][j][i].y * fac1 + ucat[k][j][i+1].y * fac2;
                        w = ucat[k][j][i].z * fac1 + ucat[k][j][i+1].z * fac2;
                        p1 = p[k][j][i] * fac1 + p[k][j][i+1] * fac2;
                        uu = sqrt( u*u + v*v + w*w );
                        fprintf(pFile1, "%le %le %le %le %le %le %le \n", z, y,
                                u, v, w, p1, nvert[k][j][i] );

                }
                }

                fclose(pFile1);

		//printf(" HERE 2 \n");

                j = iy_zx;
                for (i=xs; i<xe-1; i++){
                for (k=zs; k<ze-1; k++){
                        fac1 = (coor[k][j+1][i].y - zx_plane)/(coor[k][j+1][i].y - coor[k][j][i].y);
                        fac2 = (-coor[k][j][i].y + zx_plane)/(coor[k][j+1][i].y - coor[k][j][i].y);

                        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        u = ucat[k][j][i].x * fac1 + ucat[k][j+1][i].x * fac2;
                        v = ucat[k][j][i].y * fac1 + ucat[k][j+1][i].y * fac2;
                        w = ucat[k][j][i].z * fac1 + ucat[k][j+1][i].z * fac2;
                        p1 = p[k][j][i] * fac1 + p[k][j+1][i] * fac2;
                        uu = sqrt( u*u + v*v + w*w );
                        fprintf(pFile2, "%le %le %le %le %le %le %le \n", z, x,
                                u, v, w, p1, nvert[k][j][i] );

                }
                }

                fclose(pFile2);


		//printf(" HERE 3 \n");
         
//  		for (k=zs; k<ze-2; k++){
//                for (j=ys; j<ye-2; j++){
//                for (i=xs; i<xe-2; i++){
//			x = coor[k+1][j+1][i+1].x; y = coor[k+1][j+1][i+1].y; z = coor[k+1][j+1][i+1].z; 
//			u = ucat[k+1][j+1][i+1].x; v = ucat[k+1][j+1][i+1].y; w = ucat[k+1][j+1][i+1].z; p1 = p[k+1][j+1][i+1]; 
//			uu = sqrt( u*u + v*v + w*w );
//			fprintf(pFile, "%le %le %le %le %le %le %le %le \n", x, y, z,
  //                      	u, v, w, p1, nvert[k][j][i] );
//
//		}
//		}
//		}
//
//		fclose(pFile);
/*		
//		if(only_V==2) {	// Z Vorticity
		double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
                double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                double csi0, csi1, csi2;
                double eta0, eta1, eta2;
                double zet0, zet1, zet2;
                double ajc;

	
		for (k=zs+1; k<ze-2; k++)
		for (j=ys+1; j<ye-2; j++)
		for (i=xs+1; i<xe-2; i++) {

//				printf(" I AM HEER 0!!");
			csi0 = csi[k][j][i].x; csi1 = csi[k][j][i].y; csi2 = csi[k][j][i].z;
			eta0= eta[k][j][i].x; eta1 = eta[k][j][i].y; eta2 = eta[k][j][i].z;
			zet0 = zet[k][j][i].x; zet1 = zet[k][j][i].y; zet2 = zet[k][j][i].z;
			ajc = aj[k][j][i];
		        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
//				printf(" I AM HEER 1!!");
			if(i==0 || j==0 || k==0) vor_1 = 0., vor_2=0., vor_3=0.;
			else {
				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
			}
				
//				printf(" I AM HEER 2 !!");
			vor_1 = dw_dy - dv_dz; vor_2 = du_dz - dw_dx; vor_3 = dv_dx - du_dy;
                        fprintf(pFile1, "%le %le %le %le %le %le %le  \n", x, y, z,
                                vor_1, vor_2, vor_3,  nvert[k][j][i] );


		}
			
//		}
		
		fclose(pFile1);
*/

//  		DAVecRestoreArray(user[bi].fda, user[bi].Csi, &csi);
//  		DAVecRestoreArray(user[bi].fda, user[bi].Eta, &eta);
//  		DAVecRestoreArray(user[bi].fda, user[bi].Zet, &zet);
//  		DAVecRestoreArray(user[bi].da, user[bi].Aj, &aj);


		DAVecRestoreArray(user[bi].fda, user[bi].Ucat, &ucat);
		DAVecRestoreArray(user[bi].da, user[bi].Nvert, &nvert);
                DAVecRestoreArray(user[bi].da, user[bi].P, &p);

	}

//        DAVecRestoreArray(user->fda, user->Csi, &csi);
//        DAVecRestoreArray(user->fda, user->Eta, &eta);
//        DAVecRestoreArray(user->fda, user->Zet, &zet);
//        DAVecRestoreArray(user->da, user->Aj, &aj);

	return 0;
}


// TEC output in ASCII format


PetscErrorCode TECIOOut_V(UserCtx *user, int only_V)	// seokkoo
{
	PetscInt bi;
	PetscInt IMax, JMax, KMax;
        PetscReal x, y, z, u, v, w, p1;
	char filen[80];
	sprintf(filen, "%sResult%06d.plt", prefix, ti);

        FILE * pFile;
        pFile = fopen(filen, "w");
        
	FILE * pFile1;	
	sprintf(filen, "%sResult-Vorticity%06d.plt", prefix, ti);
        pFile1 = fopen(filen, "w");


        fprintf(pFile, " TITLE = \"Flow Field\" \n");
        fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"U\", \"V\", \"W\", \"P\", \"nv\" \n" );

        fprintf(pFile1, " TITLE = \"Vorticity Field\" \n");
        fprintf(pFile1, " VARIABLES = \"X\", \"Y\", \"Z\", \"vor-1\", \"vor-2\", \"vor-3\", \"nv\" \n" );

//        PetscReal       ***aj;
//        Cmpnts  ***csi, ***eta, ***zet;

//  	DAVecGetArray(user->fda, user->Csi, &csi);
//  	DAVecGetArray(user->fda, user->Eta, &eta);
//  	DAVecGetArray(user->fda, user->Zet, &zet);
//  	DAVecGetArray(user->da, user->Aj, &aj);


	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;
    
		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o, ***csi, ***eta, ***zet;
		PetscReal	***p, ***nvert, ***level;
		Vec		Coor, zCoor, nCoor;
		VecScatter	ctx;
		Vec K_Omega;
		PetscReal	uu;	
		PetscReal	vor_1, vor_2, vor_3;	

		DAVecGetArray(user[bi].da, user[bi].Nvert, &nvert);
		DAVecGetArray(user[bi].fda, user[bi].Ucat, &ucat);
		DAVecGetArray(user[bi].da, user[bi].P, &p);


  		DAVecGetArray(user[bi].fda, user[bi].Csi, &csi);
  		DAVecGetArray(user[bi].fda, user[bi].Eta, &eta);
  		DAVecGetArray(user[bi].fda, user[bi].Zet, &zet);
  		DAVecGetArray(user[bi].da, user[bi].Aj, &aj);

		
		IMax = mx-2;
		JMax = my-2;
		KMax = mz-2;

		fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );
		
		fprintf(pFile1, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax-1, JMax-1, KMax-1 );

		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);
                 
  		for (k=zs; k<ze-2; k++){
                for (j=ys; j<ye-2; j++){
                for (i=xs; i<xe-2; i++){
			x = coor[k+1][j+1][i+1].x; y = coor[k+1][j+1][i+1].y; z = coor[k+1][j+1][i+1].z; 
			u = ucat[k+1][j+1][i+1].x; v = ucat[k+1][j+1][i+1].y; w = ucat[k+1][j+1][i+1].z; p1 = p[k+1][j+1][i+1]; 
			uu = sqrt( u*u + v*v + w*w );
			fprintf(pFile, "%le %le %le %le %le %le %le %le \n", x, y, z,
                        	u, v, w, p1, nvert[k][j][i] );

		}
		}
		}

		fclose(pFile);
		
//		if(only_V==2) {	// Z Vorticity
		double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
                double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
                double csi0, csi1, csi2;
                double eta0, eta1, eta2;
                double zet0, zet1, zet2;
                double ajc;

	
		for (k=zs+1; k<ze-2; k++)
		for (j=ys+1; j<ye-2; j++)
		for (i=xs+1; i<xe-2; i++) {

//				printf(" I AM HEER 0!!");
			csi0 = csi[k][j][i].x; csi1 = csi[k][j][i].y; csi2 = csi[k][j][i].z;
			eta0= eta[k][j][i].x; eta1 = eta[k][j][i].y; eta2 = eta[k][j][i].z;
			zet0 = zet[k][j][i].x; zet1 = zet[k][j][i].y; zet2 = zet[k][j][i].z;
			ajc = aj[k][j][i];
		        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
//				printf(" I AM HEER 1!!");
			if(i==0 || j==0 || k==0) vor_1 = 0., vor_2=0., vor_3=0.;
			else {
				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
			}
				
//				printf(" I AM HEER 2 !!");
			vor_1 = dw_dy - dv_dz; vor_2 = du_dz - dw_dx; vor_3 = dv_dx - du_dy;
                        fprintf(pFile1, "%le %le %le %le %le %le %le  \n", x, y, z,
                                vor_1, vor_2, vor_3,  nvert[k][j][i] );


		}
			
//		}
		
		fclose(pFile1);


  		DAVecRestoreArray(user[bi].fda, user[bi].Csi, &csi);
  		DAVecRestoreArray(user[bi].fda, user[bi].Eta, &eta);
  		DAVecRestoreArray(user[bi].fda, user[bi].Zet, &zet);
  		DAVecRestoreArray(user[bi].da, user[bi].Aj, &aj);


		DAVecRestoreArray(user[bi].fda, user[bi].Ucat, &ucat);
		DAVecRestoreArray(user[bi].da, user[bi].Nvert, &nvert);
                DAVecRestoreArray(user[bi].da, user[bi].P, &p);

	}

//        DAVecRestoreArray(user->fda, user->Csi, &csi);
//        DAVecRestoreArray(user->fda, user->Eta, &eta);
//        DAVecRestoreArray(user->fda, user->Zet, &zet);
//        DAVecRestoreArray(user->da, user->Aj, &aj);

	return 0;
}



PetscErrorCode TECIOOut_Averaging(UserCtx *user, IBMNodes *ibm)	// seokkoo
{
	PetscInt bi;
	PetscReal x, y, z, u, v, w, uu, vv, ww, uv, vw, wu, p1, vel_mag, pp, vortx, vorty, vortz, vortx2, vorty2, vortz2;

	double N = (double)tie + 1.0;


//	double *u_ik, *v_ik, *w_ik, *p_ik, *uu_ik_reynolds, *vv_ik_reynolds, *ww_ik_reynolds, *uv_ik_reynolds, *vw_ik_reynolds, *wu_ik_reynolds, 
//	       *uu_ik_dispersive, *vv_ik_dispersive, *ww_ik_dispersive, *uv_ik_dispersive, *vw_ik_dispersive, *wu_ik_dispersive, *nu_t_ik;

//	double *uu_reynolds, *vv_reynolds, *ww_reynolds, *uv_reynolds, *vw_reynolds, *wu_reynolds,
//               *uu_dispersive, *vv_dispersive, *ww_dispersive, *uv_dispersive, *vw_dispersive, *wu_dispersive;

		PetscPrintf(PETSC_COMM_WORLD, "HERE 33 !\n");
 

	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;
    
		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o;
		Cmpnts	***u2sum, ***u1sum,  ***usum;
		PetscReal	***p, ***nvert;
		Vec		Coor, zCoor, nCoor;
		PetscReal	***nutsum;
		//VecScatter ctx;

		PetscPrintf(PETSC_COMM_WORLD, "HERE 22 !\n");
        	double u_ik[my], v_ik[my], w_ik[my], p_ik[my], uu_ik_reynolds[my], vv_ik_reynolds[my], ww_ik_reynolds[my], uv_ik_reynolds[my], vw_ik_reynolds[my], wu_ik_reynolds[my], uu_ik_dispersive[my], vv_ik_dispersive[my], ww_ik_dispersive[my], uv_ik_dispersive[my], vw_ik_dispersive[my], wu_ik_dispersive[my], nu_t_ik[my], vw_ik_mreynolds[my];

		PetscPrintf(PETSC_COMM_WORLD, "HERE 44 %3d %3d %3d !\n", mz,my,mx);
//		double uu_reynolds[10][10][10], vv_reynolds[10][10][10], ww_reynolds[10][10][10]; //,uv_reynolds[mz][my][mx], vw_reynolds[mz][my][mx], wu_reynolds[mz][my][mx];
//Segm err      	double uu_reynolds[mz][my][mx], vv_reynolds[mz][my][mx], ww_reynolds[mz][my][mx], uv_reynolds[mz][my][mx], vw_reynolds[mz][my][mx], wu_reynolds[mz][my][mx], uu_dispersive[mz][my][mx], vv_dispersive[mz][my][mx], ww_dispersive[mz][my][mx], uv_dispersive[mz][my][mx], vw_dispersive[mz][my][mx], wu_dispersive[mz][my][mx];


		PetscPrintf(PETSC_COMM_WORLD, "HERE 55 !\n");
        	lxs = xs; lxe = xe;
        	lys = ys; lye = ye;
        	lzs = zs; lze = ze;

        	if (xs==0) lxs = xs+1;
        	if (ys==0) lys = ys+1;
        	if (zs==0) lzs = zs+1;

        	if (xe==mx) lxe = xe-1;
        	if (ye==my) lye = ye-1;
        	if (ze==mz) lze = ze-1;

		
		PetscInt IMax, JMax, KMax;

                Vec P_sum;
                PetscReal ***psum;
                Vec P_square_sum;
                PetscReal ***p2sum;
                Vec Vort_sum, Vort_square_sum;
                Cmpnts ***vortsum, ***vort2sum;
			
	        Vec Nut_sum;


		Vec T_sum, TT_sum, TU_sum, TP_sum, Prt_sum;
	
		PetscReal ***t_sum, ***tt_sum, ***tp_sum, ***prt_sum; 

		Cmpnts ***tu_sum;



		IMax = mx-2; JMax = my-2; KMax = mz-2;

		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);

                PetscViewer viewer;
                char filen1[128];
		if (averaging_option >= 1){

                        DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
                        DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_cross_sum);
                        DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);
                        DACreateGlobalVector(user[bi].da, &P_sum);

			if (LES) DACreateGlobalVector(user[bi].da, &Nut_sum);


                        sprintf(filen1, "su0_%06d_%1d.dat", ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, (user[bi].Ucat_sum));
                        PetscViewerDestroy(viewer);

                        sprintf(filen1, "su1_%06d_%1d.dat", ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, (user[bi].Ucat_cross_sum));
                        PetscViewerDestroy(viewer);

                        sprintf(filen1, "su2_%06d_%1d.dat", ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, (user[bi].Ucat_square_sum));
                        PetscViewerDestroy(viewer);

                        sprintf(filen1, "sp_%06d_%1d.dat", ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, P_sum);
                        PetscViewerDestroy(viewer);

			if (LES) { 
                       	 sprintf(filen1, "snut_%06d_%1d.dat", ti, user[bi]._this);
                       	 PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
                       	 VecLoadIntoVector(viewer, Nut_sum);
                       	 PetscViewerDestroy(viewer);
			}

                        DAVecGetArray(user[bi].fda, user[bi].Ucat_sum, &usum);
                        DAVecGetArray(user[bi].fda, user[bi].Ucat_cross_sum, &u1sum);
                        DAVecGetArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);
                        DAVecGetArray(user[bi].da, P_sum, &psum);

                        if (LES) DAVecGetArray(user[bi].da, Nut_sum, &nutsum);

		}

		if(averaging_option>=2) {

			DACreateGlobalVector(user[bi].da, &P_square_sum);

			sprintf(filen1, "sp2_%06d_%1d.dat", ti, user[bi]._this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, P_square_sum);
			PetscViewerDestroy(viewer);

			DAVecGetArray(user[bi].da, P_square_sum, &p2sum);
		}

		if(averaging_option>=3) {

			DACreateGlobalVector(user[bi].fda, &Vort_sum);
			DACreateGlobalVector(user[bi].fda, &Vort_square_sum);

			sprintf(filen1, "svo_%06d_%1d.dat", ti, user[bi]._this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, Vort_sum);
			PetscViewerDestroy(viewer);

			sprintf(filen1, "svo2_%06d_%1d.dat", ti, user[bi]._this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen1, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, Vort_square_sum);
			PetscViewerDestroy(viewer);

			DAVecGetArray(user[bi].fda, Vort_sum, &vortsum);
			DAVecGetArray(user[bi].fda, Vort_square_sum, &vort2sum);
                }
/*
                u_ik = (double*) malloc (my);
                v_ik = (double*) malloc (my);
                w_ik = (double*) malloc (my);
                p_ik = (double*) malloc (my);
                nu_t_ik = (double*) malloc (my);

                uu_ik_reynolds = (double*) malloc (my);
                vv_ik_reynolds = (double*) malloc (my);
                ww_ik_reynolds = (double*) malloc (my);
                uv_ik_reynolds = (double*) malloc (my);
                vw_ik_reynolds = (double*) malloc (my);
                wu_ik_reynolds = (double*) malloc (my);

                uu_ik_dispersive = (double*) malloc (my);
                vv_ik_dispersive = (double*) malloc (my);
                ww_ik_dispersive = (double*) malloc (my);
                uv_ik_dispersive = (double*) malloc (my);
                vw_ik_dispersive = (double*) malloc (my);
                wu_ik_dispersive = (double*) malloc (my);

                
		uu_reynolds = (double*) malloc (mx*my*mz);
		vv_reynolds = (double*) malloc (mx*my*mz);
	 	ww_reynolds = (double*) malloc (mx*my*mz);
		uv_reynolds = (double*) malloc (mx*my*mz);
		vw_reynolds = (double*) malloc (mx*my*mz);
		wu_reynolds = (double*) malloc (mx*my*mz);

                uu_dispersive = (double*) malloc (mx*my*mz);
                vv_dispersive = (double*) malloc (mx*my*mz);
                ww_dispersive = (double*) malloc (mx*my*mz);
                uv_dispersive = (double*) malloc (mx*my*mz);
                vw_dispersive = (double*) malloc (mx*my*mz);
                wu_dispersive = (double*) malloc (mx*my*mz);
*/

/*
                for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
			x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N;

                        uu_reynolds[k][j][i] = ( u2sum[k][j][i].x/N - u*u );
                        vv_reynolds[k][j][i] = ( u2sum[k][j][i].y/N - v*v );
                        ww_reynolds[k][j][i] = ( u2sum[k][j][i].z/N - w*w );

//                        uv_reynolds[k][j][i] = ( u1sum[k][j][i].x/N - u*v );
//                        vw_reynolds[k][j][i] = ( u1sum[k][j][i].y/N - v*w );
//                        wu_reynolds[k][j][i] = ( u1sum[k][j][i].z/N - w*u );

		}
		}
		}
*/
		double A_sum, A;
                for (j=ys; j<ye; j++) {
			u_ik[j] = 0.0; v_ik[j] = 0.0; w_ik[j] = 0.0; p_ik[j] = 0.0; nu_t_ik[j] = 0.0; 
			uu_ik_reynolds[j] = 0.0; vv_ik_reynolds[j] = 0.0; ww_ik_reynolds[j] = 0.0;
                        uv_ik_reynolds[j] = 0.0; vw_ik_reynolds[j] = 0.0; wu_ik_reynolds[j] = 0.0;
			A_sum = 0.0;
                	for (k=lzs; k<lze; k++) {
                	for (i=lxs; i<lxe; i++) {
				u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N; p1 = psum[k][j][i] / N;
				A = (coor[k][j][i].z-coor[k-1][j][i].z)*(coor[k][j][i].x-coor[k][j][i-1].x); // just for cartesian grid	
				u_ik[j] += u*A;  v_ik[j] += v*A; w_ik[j] += w*A; p_ik[j] += p1*A; 
				if (LES) nu_t_ik[j] += nutsum[k][j][i]*A/N;

	                        double uu_reynolds = ( u2sum[k][j][i].x/N - u*u );
        	                double vv_reynolds = ( u2sum[k][j][i].y/N - v*v );
                	        double ww_reynolds = ( u2sum[k][j][i].z/N - w*w );

                        	double uv_reynolds = ( u1sum[k][j][i].x/N - u*v );
	                        double vw_reynolds = ( u1sum[k][j][i].y/N - v*w );
        	                double wu_reynolds = ( u1sum[k][j][i].z/N - w*u );

				uu_ik_reynolds[j] += uu_reynolds*A; 
                                vv_ik_reynolds[j] += vv_reynolds*A;
                                ww_ik_reynolds[j] += ww_reynolds*A;
                                uv_ik_reynolds[j] += uv_reynolds*A;
                                vw_ik_reynolds[j] += vw_reynolds*A;
                                wu_ik_reynolds[j] += wu_reynolds*A;
				
				A_sum += A;
				
			}
			}

			u_ik[j] /= A_sum;  v_ik[j] /= A_sum; w_ik[j] /= A_sum; p_ik[j] /= A_sum; nu_t_ik[j] /= A_sum;

                        uu_ik_reynolds[j] /= A_sum;
                        vv_ik_reynolds[j] /= A_sum;
                        ww_ik_reynolds[j] /= A_sum;
                        uv_ik_reynolds[j] /= A_sum;
                        vw_ik_reynolds[j] /= A_sum;
                        wu_ik_reynolds[j] /= A_sum;

		}

/*
                for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                        u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N;
                	uu_dispersive[k][j][i] = (u-u_ik[j])*(u-u_ik[j]);
                	vv_dispersive[k][j][i] = (v-v_ik[j])*(v-v_ik[j]);
                	ww_dispersive[k][j][i] = (w-w_ik[j])*(w-w_ik[j]);
                	uv_dispersive[k][j][i] = (u-u_ik[j])*(v-v_ik[j]);
                	vw_dispersive[k][j][i] = (v-v_ik[j])*(w-w_ik[j]);
                	wu_dispersive[k][j][i] = (w-w_ik[j])*(u-u_ik[j]);
		}
		}
		}
*/

                for (j=lys; j<lye; j++) {
                        uu_ik_dispersive[j] = 0.0; vv_ik_dispersive[j] = 0.0; ww_ik_dispersive[j] = 0.0;
                        uv_ik_dispersive[j] = 0.0; vw_ik_dispersive[j] = 0.0; wu_ik_dispersive[j] = 0.0;
                        A_sum = 0.0;
                        for (k=lzs; k<lze; k++) {
                        for (i=lxs; i<lxe; i++) {
                                A = (coor[k][j][i].z-coor[k-1][j][i].z)*(coor[k][j][i].x-coor[k][j][i-1].x); // just for cartesian grid
                        	u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N;

	                        double uu_dispersive = (u-u_ik[j])*(u-u_ik[j]);
        	                double vv_dispersive = (v-v_ik[j])*(v-v_ik[j]);
                	        double ww_dispersive = (w-w_ik[j])*(w-w_ik[j]);
                        	double uv_dispersive = (u-u_ik[j])*(v-v_ik[j]);
	                        double vw_dispersive = (v-v_ik[j])*(w-w_ik[j]);
        	                double wu_dispersive = (w-w_ik[j])*(u-u_ik[j]);



                                uu_ik_dispersive[j] += uu_dispersive*A;
                                vv_ik_dispersive[j] += vv_dispersive*A;
                                ww_ik_dispersive[j] += ww_dispersive*A;
                                uv_ik_dispersive[j] += uv_dispersive*A;
                                vw_ik_dispersive[j] += vw_dispersive*A;
                                wu_ik_dispersive[j] += wu_dispersive*A;

                                A_sum += A;

                        }
                        }

                        uu_ik_dispersive[j] /= A_sum;
                        vv_ik_dispersive[j] /= A_sum;
                        ww_ik_dispersive[j] /= A_sum;
                        uv_ik_dispersive[j] /= A_sum;
                        vw_ik_dispersive[j] /= A_sum;
                        wu_ik_dispersive[j] /= A_sum;

                }

//		PetscPrintf(PETSC_COMM_WORLD, "HERE 11 !\n");

		char filen[80];
        	FILE * pFile;
/*
		sprintf(filen, "%sResults%06d-avg.plt", prefix, ti);
        	pFile = fopen(filen, "w");
        	fprintf(pFile, " TITLE = \"Averaging velocity\" \n");
		fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"U\", \"V\", \"W\", \"P\", \"Nu_t\"\n" );
                fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );
		
		for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
			x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
			u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N;
                        p1 = (psum[k][j][i] / N);
					
   			fprintf(pFile, "%le %le %le %le %le %le %le %le \n", x, y, z, u, v, w, p1, nutsum[k][j][i]/N );
              	}
              	}
		}
		fclose(pFile);


                sprintf(filen, "%sResults-Reynolds-stress%06d-avg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                fprintf(pFile, " TITLE = \"Reynolds stress\" \n");
                fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"uu_reynolds\", \"vv_reynolds\", \"ww_reynolds\", \"uv_reynolds\", \"vw_reynolds\", \"wu_reynolds\" \"k\", \n" );
                fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );

                for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        fprintf(pFile, "%le %le %le %le %le %le %le %le %le %le \n", x, y, z, uu_reynolds[k][j][i], vv_reynolds[k][j][i], ww_reynolds[k][j][i], uv_reynolds[k][j][i], vw_reynolds[k][j][i], wu_reynolds[k][j][i], 0.5*(uu_reynolds[k][j][i] + vv_reynolds[k][j][i] + ww_reynolds[k][j][i]) );
                }
                }
                }
                fclose(pFile);

                sprintf(filen, "%sResults-dispersive-stress%06d-avg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                fprintf(pFile, " TITLE = \"Dispersive stress\" \n");
                fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"uu_dispersive\", \"vv_dispersive\", \"ww_dispersvie\", \"uv_dispersive\", \"vw_dispersive\", \"wu_dispersive\" \n" );
                fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );

                for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        fprintf(pFile, "%le %le %le %le %le %le %le %le %le \n", x, y, z, uu_dispersive[k][j][i], vv_dispersive[k][j][i], ww_dispersive[k][j][i], uv_dispersive[k][j][i], vw_dispersive[k][j][i], wu_dispersive[k][j][i] );
                }
                }
                }
                fclose(pFile);
*/
                sprintf(filen, "%sResults%06d-ikavg.plt", prefix, ti);
                pFile = fopen(filen, "w");
		i = 1; k = 1;
                for (j=lys; j<lye; j++) {
                        y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y); 
                        fprintf(pFile, "%le %le %le %le %le \n", y, u_ik[j], v_ik[j], w_ik[j], p_ik[j] );
                }
                fclose(pFile);


                sprintf(filen, "%sResults-Reynolds-stress%06d-ikavg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                i = 1; k = 1;
                for (j=lys; j<lye; j++) {
                        y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y);
                        fprintf(pFile, "%le %le %le %le %le %le %le %le %le \n", y, uu_ik_reynolds[j], vv_ik_reynolds[j], ww_ik_reynolds[j], uv_ik_reynolds[j], vw_ik_reynolds[j], wu_ik_reynolds[j], 0.5 * (uu_ik_reynolds[j] + vv_ik_reynolds[j] + ww_ik_reynolds[j]), w_ik[j] );
                }
                fclose(pFile);

                sprintf(filen, "%sResults-dispersive-stress%06d-ikavg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                i = 1; k = 1;
                for (j=lys; j<lye; j++) {
                        y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y);
                        fprintf(pFile, "%le %le %le %le %le %le %le \n", y, uu_ik_dispersive[j], vv_ik_dispersive[j], ww_ik_dispersive[j], uv_ik_dispersive[j], vw_ik_dispersive[j], wu_ik_dispersive[j] );
                }
                fclose(pFile);


                sprintf(filen, "%sResults-reynolds-dispersive-stress-vw%06d-ikavg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                i = 1; k = 1;
		double viscous_ik, viscous_nut, dwdy;
		double nu1, nut1;
		double w1, w2;		

                for (j=lys; j<lye-1; j++) {
                        y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y);
			nu1 = 1.0/user->ren;
			if (LES) nut1 = nutsum[k][j][i]/N; //nu_t_ik[j];
			
			w1 = usum[k][j][i].z / N;
			w2 = usum[k][j+1][i].z / N;

			dwdy = (w2-w1)/(coor[k][j+1][i].y-coor[k][j][i].y);
			//if (j==1)  dwdy = (w_ik[j+1]-w_ik[j])/(coor[k][j+1][i].y-coor[k][j][i].y);
                        //if (j==my-2)  dwdy = (w_ik[j]-w_ik[j-1])/(coor[k][j][i].y-coor[k][j-1][i].y);
			viscous_ik = nu1 * dwdy;
			viscous_nut = 0.0;

			if (LES) viscous_nut = nut1 * dwdy;
			
                        fprintf(pFile, "%le %le %le %le %le \n", y, -vw_ik_dispersive[j], -vw_ik_reynolds[j], viscous_ik + viscous_nut, w_ik[j]  );
                }
                fclose(pFile);

		if (LES) {
                sprintf(filen, "%sResults-nu_t%06d-ikavg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                i = 1; k = 1;
                for (j=lys; j<lye; j++) {
                        y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y);
                        fprintf(pFile, "%le %le \n", y, nu_t_ik[j] );
                }
                fclose(pFile);
		}

/*
//		Calculate different terms in the time averaged equation
                double dwu_dx_avg[mz][my][mx],  dwv_dy_avg[mz][my][mx], dww_dz_avg[mz][my][mx],
                       dwu_dx_fluc[mz][my][mx],  dwv_dy_fluc[mz][my][mx], dww_dz_fluc[mz][my][mx],
                       dw_dx2[mz][my][mx], dw_dy2[mz][my][mx],dw_dz2[mz][my][mx],
		       dwu_dx_dispersive[mz][my][mx],  dwv_dy_dispersive[mz][my][mx], dww_dz_dispersive[mz][my][mx];
		

		double val_left, val_right, val_upper, val_low, val_front, val_back, dh;
		int ip1, im1, jp1, jm1, kp1, km1;
                for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
			ip1 = i+1; im1 = i-1;
			if (i == lxs) im1 = i; if (i == lxe-1) ip1 = i;
			val_left =  usum[k][j][im1].z * usum[k][j][im1].x / N / N;
			val_right =  usum[k][j][ip1].z * usum[k][j][ip1].x / N / N;
			dh = 1.0/ (coor[k][j][ip1].x - coor[k][j][im1].x);
			dwu_dx_avg[k][j][i] = -(val_right - val_left) * dh;

                        val_left =  wu_reynolds[k][j][im1];
                        val_right =  wu_reynolds[k][j][ip1];;
                        dwu_dx_fluc[k][j][i] = -(val_right - val_left) * dh;

                        val_left =  wu_dispersive[k][j][im1];
                        val_right =  wu_dispersive[k][j][ip1];;
                        dwu_dx_dispersive[k][j][i] = -(val_right - val_left) * dh;


                        jp1 = j+1; jm1 = j-1;
                        if (j == lys) jm1 = j; if (j == lye-1) jp1 = j;
                        val_low =  usum[k][jm1][i].z * usum[k][jm1][i].y / N / N;
                        val_upper =  usum[k][jp1][i].z * usum[k][jp1][i].y / N / N;
                        dh = 1.0/ (coor[k][jp1][i].y - coor[k][jm1][i].y);
                        dwv_dy_avg[k][j][i] = -(val_upper - val_low) * dh;

                        val_low =  vw_reynolds[k][jm1][i];
                        val_upper =  vw_reynolds[k][jp1][i];
                        dwv_dy_fluc[k][j][i] = -(val_upper - val_low) * dh;

                        val_low =  vw_dispersive[k][jm1][i];
                        val_upper =  vw_dispersive[k][jp1][i];
                        dwv_dy_dispersive[k][j][i] = -(val_upper - val_low) * dh;


                        kp1 = k+1; km1 = k-1;
                        if (k == lzs) km1 = k; if (k == lze-1) kp1 = k;
                        val_back =  usum[km1][j][i].z * usum[km1][j][i].z / N / N;
                        val_front =  usum[kp1][j][i].z * usum[kp1][j][i].z / N / N;
                        dh = 1.0/ (coor[kp1][j][i].z - coor[km1][j][i].z);
                        dww_dz_avg[k][j][i] = -(val_front - val_back) * dh;


                        val_back = ww_reynolds[km1][j][i];
                        val_front = ww_reynolds[kp1][j][i];
                        dww_dz_fluc[k][j][i] = -(val_front - val_back) * dh;	

                        val_back = ww_dispersive[km1][j][i];
                        val_front = ww_dispersive[kp1][j][i];
                        dww_dz_dispersive[k][j][i] = -(val_front - val_back) * dh;
					
		}
		}
		}		
*/
/*
                sprintf(filen, "%sResults-buget%06d-avg.plt", prefix, ti);
                pFile = fopen(filen, "w");
                fprintf(pFile, " TITLE = \"buget\" \n");
                fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"dwu_dx_avg\", \"dwv_dy_avg\", \"dww_dz_avg\", \"dwu_dx_Reynolds\", \"dwv_dy_Reynolds\", \"dww_dz_Reynolds\",  \"dwu_dx_dispersive\", \"dwv_dy_dispersive\", \"dww_dz_dispersive\" \n" );
                fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );

                for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                        x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
                        fprintf(pFile, "%le %le %le %le %le %le %le %le %le %le %le %le \n", x, y, z, dwu_dx_avg[k][j][i], dwv_dy_avg[k][j][i], dww_dz_avg[k][j][i], dwu_dx_fluc[k][j][i], dwv_dy_fluc[k][j][i], dww_dz_fluc[k][j][i],  dwu_dx_dispersive[k][j][i], dwv_dy_dispersive[k][j][i], dww_dz_dispersive[k][j][i] );
                }
                }
                }
                fclose(pFile);
*/


		// 
		if (rotor_model) {

			PetscPrintf(PETSC_COMM_WORLD, "Rotor Averaging !\n");	

			Vec v_eul;
			
			char fname[80];

 			Cmpnts  ***veul;

                	DACreateGlobalVector(user[bi].fda, &v_eul);
			DAVecGetArray(user[bi].fda, v_eul, &veul);

                	for (k=lzs; k<lze; k++) {
                	for (j=lys; j<lye; j++) {
                	for (i=lxs; i<lxe; i++) {
	                       u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N;

        	               	double uu_reynolds = ( u2sum[k][j][i].x/N - u*u );
                	       	double vv_reynolds = ( u2sum[k][j][i].y/N - v*v );
                        	double ww_reynolds = ( u2sum[k][j][i].z/N - w*w );

                        	veul[k][j][i].x = uu_reynolds;
                        	veul[k][j][i].y = vv_reynolds;
                        	veul[k][j][i].z = ww_reynolds;

                	}
                	}
                	}


			DAVecRestoreArray(user[bi].fda, v_eul, &veul);

			sprintf(fname, "uu_disk.plt");
			PetscPrintf(PETSC_COMM_WORLD, "uu Interpolation !\n");
 			Calc_U_lagr(&(user[bi]), ibm, v_eul, fname);
			PetscPrintf(PETSC_COMM_WORLD, "uu Interpolation !\n");

			DAVecGetArray(user[bi].fda, v_eul, &veul);
                	for (k=lzs; k<lze; k++) {
                	for (j=lys; j<lye; j++) {
                	for (i=lxs; i<lxe; i++) {

                        	veul[k][j][i].x = usum[k][j][i].x / N; veul[k][j][i].y = usum[k][j][i].y / N; veul[k][j][i].z = usum[k][j][i].z / N;

                	}
                	}
                	}

			sprintf(fname, "U_disk.plt");

			PetscPrintf(PETSC_COMM_WORLD, "U Interpolation !\n");
 			Calc_U_lagr(&(user[bi]), ibm, v_eul, fname);

			PetscPrintf(PETSC_COMM_WORLD, "U Interpolation !\n");
			DAVecRestoreArray(user[bi].fda, v_eul, &veul);
		}



		if (temperature) {

                        DACreateGlobalVector(user[bi].da, &T_sum);
                        DACreateGlobalVector(user[bi].da, &TT_sum);
                        DACreateGlobalVector(user[bi].fda, &TU_sum);
                        DACreateGlobalVector(user[bi].da, &TP_sum);

			sprintf(path, ".");
			if (les_prt) DACreateGlobalVector(user[bi].da, &Prt_sum);

                        sprintf(filen, "%s/st_%06d_%1d.dat", path, ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, T_sum);
                        PetscViewerDestroy(viewer);

                        sprintf(filen, "%s/st2_%06d_%1d.dat", path, ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, TT_sum);
                        PetscViewerDestroy(viewer);

                        sprintf(filen, "%s/stu_%06d_%1d.dat", path, ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, TU_sum);
                        PetscViewerDestroy(viewer);

                        sprintf(filen, "%s/stp_%06d_%1d.dat", path, ti, user[bi]._this);
                        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
                        VecLoadIntoVector(viewer, TP_sum);
                        PetscViewerDestroy(viewer);

                        if(les_prt) {
                                sprintf(filen, "%s/sprt_%06d_%1d.dat", path, ti, user[bi]._this);
                                PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
                                VecLoadIntoVector(viewer, Prt_sum);
                                PetscViewerDestroy(viewer);
                        }


			DAVecGetArray(user[bi].da, T_sum, &t_sum);
			DAVecGetArray(user[bi].da, TT_sum, &tt_sum);
			DAVecGetArray(user[bi].fda, TU_sum, &tu_sum);
			DAVecGetArray(user[bi].da, TP_sum, &tp_sum);
			if(les_prt) DAVecGetArray(user[bi].da, Prt_sum, &prt_sum);


			double t_ik[my], tt_ik[my], tu_ik[my], tv_ik[my], tw_ik[my], tp_ik[my], prt_ik[my];
		
			double tt_re[mz][my][mx], tu_re[mz][my][mx], tv_re[mz][my][mx],	tw_re[mz][my][mx], tp_re[mz][my][mx];

	                for (k=lzs; k<lze; k++) {
        	        for (j=lys; j<lye; j++) {
                	for (i=lxs; i<lxe; i++) {
                        	x = coor[k][j][i].x; y = coor[k][j][i].y; z = coor[k][j][i].z;
	                        u = usum[k][j][i].x / N; v = usum[k][j][i].y / N; w = usum[k][j][i].z / N;
				double tmprt = t_sum[k][j][i]/N;
				double p = psum[k][j][i]/N;

        	                tt_re[k][j][i] = ( tt_sum[k][j][i]/N - tmprt*tmprt );
                	        tu_re[k][j][i] = ( tu_sum[k][j][i].x/N - tmprt*u );
                        	tv_re[k][j][i] = ( tu_sum[k][j][i].y/N - tmprt*v );
                        	tw_re[k][j][i] = ( tu_sum[k][j][i].z/N - tmprt*w );
                        	tp_re[k][j][i] = ( tp_sum[k][j][i]/N - tmprt*p );

	                }
        	        }
                	}


			double A_sum, A;
	                for (j=ys; j<ye; j++) {
				t_ik[j] = 0.0; tt_ik[j] = 0.0; tu_ik[j] = 0.0; tv_ik[j] = 0.0; tw_ik[j] = 0.0; 
				tp_ik[j] = 0.0; prt_ik[j] = 0.0; 
				A_sum = 0.0;
	                	for (k=lzs; k<lze; k++) {
        	        	for (i=lxs; i<lxe; i++) {
					A = (coor[k][j][i].z-coor[k-1][j][i].z)*(coor[k][j][i].x-coor[k][j][i-1].x); // just for cartesian grid	
					t_ik[j] += t_sum[k][j][i]*A/N;  
					tt_ik[j] += tt_re[k][j][i]*A; 
					tu_ik[j] += tu_re[k][j][i]*A; 
					tv_ik[j] += tv_re[k][j][i]*A; 
					tw_ik[j] += tw_re[k][j][i]*A; 
					tp_ik[j] += tp_re[k][j][i]*A; 
					if (les_prt) prt_ik[j] += prt_sum[k][j][i]*A/N;
				
					A_sum += A;
				
				}
				}

				t_ik[j] /= A_sum;  tt_ik[j] /= A_sum; tu_ik[j] /= A_sum; tv_ik[j] /= A_sum; tw_ik[j] /= A_sum; tp_ik[j] /= A_sum;
				if (les_prt) prt_ik[j] /= A_sum;

			}


                	if (les_prt) {
	                	sprintf(filen, "%sResults-pr_t%06d-ikavg.plt", prefix, ti);
        		        pFile = fopen(filen, "w");
		                i = 1; k = 1;
		                for (j=lys; j<lye; j++) {
                		        y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y);
		                        fprintf(pFile, "%le %le \n", y, prt_ik[j] );
		                }
		                fclose(pFile);
	                }


                        sprintf(filen, "%sResults-Tmprt%06d-ikavg.plt", prefix, ti);
                        pFile = fopen(filen, "w");
                        i = 1; k = 1;
                        for (j=lys; j<lye; j++) {
                                y = 0.5 * (coor[k][j][i].y + coor[k][j-1][i].y);
                                fprintf(pFile, "%le %le %le %le %le %le %le \n", y, t_ik[j], tt_ik[j], tu_ik[j], tv_ik[j], tw_ik[j], tp_ik[j] );
                        }
                        fclose(pFile);



		}






	
		DAVecRestoreArray(fda, Coor, &coor);
    
                if (averaging_option >= 1){
                        DAVecRestoreArray(user[bi].fda, user[bi].Ucat_sum, &usum);
                        DAVecRestoreArray(user[bi].fda, user[bi].Ucat_cross_sum, &u1sum);
                        DAVecRestoreArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);
			DAVecRestoreArray(user[bi].da, P_sum, &psum);
                        VecDestroy(P_sum);
			if (LES) {
                        	DAVecRestoreArray(user[bi].da, Nut_sum, &nutsum);
                        	VecDestroy(Nut_sum);
			}
                }


		if (averaging_option >= 2){
                        DAVecRestoreArray(user[bi].da, P_square_sum, &p2sum);
                        VecDestroy(P_square_sum);

		}

		if (averaging_option >= 3){
			DAVecRestoreArray(user[bi].fda, Vort_sum, &vortsum);
                        DAVecRestoreArray(user[bi].fda, Vort_square_sum, &vort2sum);
                        VecDestroy(Vort_sum);
                        VecDestroy(Vort_square_sum);
		} 






	}
	
	return 0;
}

PetscErrorCode TECIOOutQ(UserCtx *user, int Q)
{
	PetscInt bi;
	double x, y, z, q;

	char filen[80];
	sprintf(filen, "QCriteria%06d.plt", ti);

        FILE * pFile;

        pFile = fopen(filen, "w");


	if(Q==1) {
       		 fprintf(pFile, " TITLE = \"Q-Criterion\" \n");
       		 fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"Q\" \n" );
	}
	else if(Q==2) {
                 fprintf(pFile, " TITLE = \"Lambda2-Criterion\" \n");
                 fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"Lambda2\" \n" );
	}
	else if(Q==3) {
                 fprintf(pFile, " TITLE = \"Q-Criterion\" \n");
                 fprintf(pFile, " VARIABLES = \"X\", \"Y\", \"Z\", \"Q\" \n" );
	}

	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;
    
		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o;
		PetscReal	***p, ***nvert;
		Vec		Coor, zCoor, nCoor;
		VecScatter	ctx;
		PetscInt IMax, JMax, KMax;

		IMax = mx-2; JMax = my-2; KMax = mz-2;

                fprintf(pFile, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT \n", IMax, JMax, KMax );
	
		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);


		if(Q==1) {
			QCriteria(user);
			DAVecGetArray(user[bi].da, user[bi].P, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x = coor[k+1][j+1][i+1].x; y = coor[k+1][j+1][i+1].y; z = coor[k+1][j+1][i+1].z; q = p[k+1][j+1][i+1];
				fprintf(pFile, "%le %le %le %le \n", x, y, z, q);
			}
		}
		else if(Q==2) {
			Lambda2(user);
			DAVecGetArray(user[bi].da, user[bi].P, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
                                x = coor[k+1][j+1][i+1].x; y = coor[k+1][j+1][i+1].y; z = coor[k+1][j+1][i+1].z; q = p[k+1][j+1][i+1];
                                fprintf(pFile, "%le %le %le %le \n", x, y, z, q);
			}
		}
		else if(Q==3) {
			char filen2[128];
			PetscViewer	viewer;
			
			sprintf(filen2, "qfield%06d_%1d.dat", ti, user->_this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen2, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user[bi].P));
			PetscViewerDestroy(viewer);  
			
			DAVecGetArray(user[bi].da, user[bi].P, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
                                x = coor[k+1][j+1][i+1].x; y = coor[k+1][j+1][i+1].y; z = coor[k+1][j+1][i+1].z; q = p[k+1][j+1][i+1];
                                fprintf(pFile, "%le %le %le %le \n", x, y, z, q);
			}
		}
		
	}

	fclose(pFile);
	
	return 0;
}

PetscErrorCode FormMetrics(UserCtx *user)
{
  DA		cda;
  Cmpnts	***csi, ***eta, ***zet;
  PetscScalar	***aj;
  Vec		coords;
  Cmpnts	***coor;

  DA		da = user->da, fda = user->fda;
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

  PetscInt	xs, ys, zs, xe, ye, ze;
  DALocalInfo	info;

  PetscInt	mx, my, mz;
  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  PetscScalar	dxdc, dydc, dzdc, dxde, dyde, dzde, dxdz, dydz, dzdz;
  PetscInt	i, j, k, ia, ja, ka, ib, jb, kb;
  PetscInt	gxs, gxe, gys, gye, gzs, gze;
  PetscErrorCode	ierr;

  PetscReal	xcp, ycp, zcp, xcm, ycm, zcm;
  DAGetLocalInfo(da, &info);
  mx = info.mx; my = info.my; mz = info.mz;
  xs = info.xs; xe = xs + info.xm;
  ys = info.ys; ye = ys + info.ym;
  zs = info.zs; ze = zs + info.zm;

  gxs = info.gxs; gxe = gxs + info.gxm;
  gys = info.gys; gye = gys + info.gym;
  gzs = info.gzs; gze = gzs + info.gzm;

  DAGetCoordinateDA(da, &cda);
  DAVecGetArray(cda, Csi, &csi);
  DAVecGetArray(cda, Eta, &eta);
  DAVecGetArray(cda, Zet, &zet);
  ierr = DAVecGetArray(da, Aj,  &aj); CHKERRQ(ierr);

  DAGetGhostedCoordinates(da, &coords);
  DAVecGetArray(fda, coords, &coor);


  //  VecDuplicate(coords, &Cent);
  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  /* Calculating transformation metrics in i direction */
  for (k=lzs; k<lze; k++){
    for (j=lys; j<lye; j++){
      for (i=xs; i<lxe; i++){
	/* csi = de X dz */
	dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
		      coor[k  ][j-1][i  ].x - coor[k-1][j-1][i  ].x);
	dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
		      coor[k  ][j-1][i  ].y - coor[k-1][j-1][i  ].y);
	dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
		      coor[k  ][j-1][i  ].z - coor[k-1][j-1][i  ].z);
				       		    	    	 
	dxdz = 0.5 * (coor[k  ][j-1][i  ].x + coor[k  ][j  ][i  ].x -
		      coor[k-1][j-1][i  ].x - coor[k-1][j  ][i  ].x);
	dydz = 0.5 * (coor[k  ][j-1][i  ].y + coor[k  ][j  ][i  ].y -
		      coor[k-1][j-1][i  ].y - coor[k-1][j  ][i  ].y);
	dzdz = 0.5 * (coor[k  ][j-1][i  ].z + coor[k  ][j  ][i  ].z -
		      coor[k-1][j-1][i  ].z - coor[k-1][j  ][i  ].z);
	  
	csi[k][j][i].x = dyde * dzdz - dzde * dydz;
	csi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
	csi[k][j][i].z = dxde * dydz - dyde * dxdz;

	
      }
    }
  }

  // Need more work -- lg65
  /* calculating j direction metrics */
  for (k=lzs; k<lze; k++){
    for (j=ys; j<lye; j++){
      for (i=lxs; i<lxe; i++){

	/* eta = dz X de */
	dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
		      coor[k  ][j  ][i-1].x - coor[k-1][j  ][i-1].x);
	dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
		      coor[k  ][j  ][i-1].y - coor[k-1][j  ][i-1].y);
	dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
		      coor[k  ][j  ][i-1].z - coor[k-1][j  ][i-1].z);
			    		         	 		   	 
	dxdz = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
		      coor[k-1][j  ][i  ].x - coor[k-1][j  ][i-1].x);
	dydz = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
		      coor[k-1][j  ][i  ].y - coor[k-1][j  ][i-1].y);
	dzdz = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
		      coor[k-1][j  ][i  ].z - coor[k-1][j  ][i-1].z);
	  
	eta[k][j][i].x = dydz * dzdc - dzdz * dydc;
	eta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
	eta[k][j][i].z = dxdz * dydc - dydz * dxdc;

      }
    }
  }

  /* calculating k direction metrics */
  for (k=zs; k<lze; k++){
    for (j=lys; j<lye; j++){
      for (i=lxs; i<lxe; i++){
	dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x -
		      coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x);
	dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y -
		      coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y);
	dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z -
		      coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z);
			    		    	     	     	 
	dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
		      coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x);
	dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
		      coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y);
	dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
		      coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z);
	  
	zet[k][j][i].x = dydc * dzde - dzdc * dyde;
	zet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
	zet[k][j][i].z = dxdc * dyde - dydc * dxde;

	/*	if ((i==1 || i==mx-2) && j==1 && (k==1 || k==0)) {
	  PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", dxdc * dyde, dydc * dxde, dzdc);
	  PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", dxde, dyde, dzde);
	  PetscPrintf(PETSC_COMM_WORLD, "Met %e %e %e\n", zet[k][j][i].x, zet[k][j][i].y, zet[k][j][i].z);
	  }*/
	
      }
    }
  }

  /* calculating Jacobian of the transformation */
  for (k=lzs; k<lze; k++){
    for (j=lys; j<lye; j++){
      for (i=lxs; i<lxe; i++){

	if (i>0 && j>0 && k>0) {
	  dxdc = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
			 coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x -
			 coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x -
			 coor[k-1][j  ][i-1].x - coor[k-1][j-1][i-1].x);
	  dydc = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
			 coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y -
			 coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y -
			 coor[k-1][j  ][i-1].y - coor[k-1][j-1][i-1].y);
	  dzdc = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
			 coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z -
			 coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z -
			 coor[k-1][j  ][i-1].z - coor[k-1][j-1][i-1].z);

	  dxde = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x +
			 coor[k-1][j  ][i  ].x + coor[k-1][j  ][i-1].x - 
			 coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x -
			 coor[k-1][j-1][i  ].x - coor[k-1][j-1][i-1].x);
	  dyde = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y +
			 coor[k-1][j  ][i  ].y + coor[k-1][j  ][i-1].y - 
			 coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y -
			 coor[k-1][j-1][i  ].y - coor[k-1][j-1][i-1].y);
	  dzde = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z +
			 coor[k-1][j  ][i  ].z + coor[k-1][j  ][i-1].z - 
			 coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z -
			 coor[k-1][j-1][i  ].z - coor[k-1][j-1][i-1].z);

	  dxdz = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
			 coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x -
			 coor[k-1][j  ][i  ].x - coor[k-1][j-1][i  ].x -
			 coor[k-1][j  ][i-1].x - coor[k-1][j-1][i-1].x);
	  dydz = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
			 coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y -
			 coor[k-1][j  ][i  ].y - coor[k-1][j-1][i  ].y -
			 coor[k-1][j  ][i-1].y - coor[k-1][j-1][i-1].y);
	  dzdz = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
			 coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z -
			 coor[k-1][j  ][i  ].z - coor[k-1][j-1][i  ].z -
			 coor[k-1][j  ][i-1].z - coor[k-1][j-1][i-1].z);
		
	  aj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
	    dydc * (dxde * dzdz - dzde * dxdz) +
	    dzdc * (dxde * dydz - dyde * dxdz);
	  aj[k][j][i] = 1./aj[k][j][i];
	  
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


  //  PetscPrintf(PETSC_COMM_WORLD, "Local info: %d", info.mx);



  DAVecRestoreArray(cda, Csi, &csi);
  DAVecRestoreArray(cda, Eta, &eta);
  DAVecRestoreArray(cda, Zet, &zet);
  DAVecRestoreArray(da, Aj,  &aj);


  DAVecRestoreArray(cda, coords, &coor);


  VecAssemblyBegin(Csi);
  VecAssemblyEnd(Csi);
  VecAssemblyBegin(Eta);
  VecAssemblyEnd(Eta);
  VecAssemblyBegin(Zet);
  VecAssemblyEnd(Zet);
  VecAssemblyBegin(Aj);
  VecAssemblyEnd(Aj);

  PetscBarrier(PETSC_NULL);
  return 0;
}

PetscErrorCode Ucont_P_Binary_Input(UserCtx *user)
{
  PetscViewer	viewer;

  char filen2[128];

  PetscOptionsClearValue("-vecload_block_size");
  sprintf(filen2, "pfield%06d_%1d.dat", ti, user->_this);
  
  PetscViewer	pviewer;
  //Vec temp;
  PetscInt rank;
  PetscReal norm;
	
  if(file_exist(filen2))
  if(!onlyV) {
    //DACreateNaturalVector(user->da, &temp);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen2, FILE_MODE_READ, &pviewer);
	VecLoadIntoVector(pviewer, (user->P));
	VecNorm(user->P, NORM_INFINITY, &norm);
	PetscPrintf(PETSC_COMM_WORLD, "PIn %le\n", norm);
	PetscViewerDestroy(pviewer);
	//VecDestroy(temp);
  }

  if(nv_once) sprintf(filen2, "nvfield%06d_%1d.dat", 0, user->_this);
  else sprintf(filen2, "nvfield%06d_%1d.dat", ti, user->_this);
  
  if(cs) sprintf(filen2, "cs_%06d_%1d.dat", ti, user->_this);
  
  if( !nv_once || (nv_once && ti==tis) )
  {
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen2, FILE_MODE_READ, &pviewer);
	VecLoadIntoVector(pviewer, (user->Nvert));
	PetscViewerDestroy(pviewer);  
  }
	
}

PetscErrorCode Ucont_P_Binary_Input1(UserCtx *user)
{
  PetscViewer viewer;
  char filen[128];
  
  sprintf(filen, "ufield%06d_%1d.dat", ti, user->_this);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);

  PetscInt N;

  VecGetSize(user->Ucat, &N);
  PetscPrintf(PETSC_COMM_WORLD, "PPP %d\n", N);
  VecLoadIntoVector(viewer, (user->Ucat));
  PetscViewerDestroy(viewer);

  PetscBarrier(PETSC_NULL);
}

PetscErrorCode Ucont_P_Binary_Input_Averaging(UserCtx *user)
{
	PetscViewer viewer;
	char filen[128];
	/*
	sprintf(filen, "su0_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_sum));
	PetscViewerDestroy(viewer);

	sprintf(filen, "su1_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_cross_sum));
	PetscViewerDestroy(viewer);
	
	sprintf(filen, "su2_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_square_sum));
	PetscViewerDestroy(viewer);
	*/
	/*
	sprintf(filen, "sp_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->P));
	PetscViewerDestroy(viewer);
	*/
  
	if(pcr) {
		Vec Ptmp;
		VecDuplicate(user->P, &Ptmp);
		
		sprintf(filen, "pfield%06d_%1d.dat", ti, user->_this);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, user->P);
		PetscViewerDestroy(viewer);
		
		sprintf(filen, "sp_%06d_%1d.dat", ti, user->_this);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, Ptmp);
		PetscViewerDestroy(viewer);
		
		
		VecScale(Ptmp, -1./((double)tis+1.0));
		VecAXPY(user->P, 1., Ptmp);
		
		VecDestroy(Ptmp);
	}
	
	if(nv_once) sprintf(filen, "nvfield%06d_%1d.dat", 0, user->_this);
	else sprintf(filen, "nvfield%06d_%1d.dat", ti, user->_this);

	//if(cs) sprintf(filen2, "cs_%06d_%1d.dat", ti, user->_this);

	if( !nv_once || (nv_once && ti==tis) )
	  {
	    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	    VecLoadIntoVector(viewer, (user->Nvert));
	    PetscViewerDestroy(viewer);
	  }
	/*
	if( !nv_once || (nv_once && ti==tis) ) {
		sprintf(filen, "nvfield%06d_%1d.dat", ti, user->_this);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, (user->Nvert));
		PetscViewerDestroy(viewer);
	}
	*/
	PetscBarrier(PETSC_NULL);
}

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv)
{
	PetscTruth flag;
	
	DA	da, fda;
	Vec	qn, qnm;
	Vec	c;
	UserCtx	*user;

	PetscErrorCode ierr;
		
	IBMNodes	ibm, *ibm0, *ibm1, *wtm;

	PetscInitialize(&argc, &argv, (char *)0, help);

	
	PetscOptionsInsertFile(PETSC_COMM_WORLD, "control.dat", PETSC_TRUE);
	
	
	char tmp_str[256];
	PetscOptionsGetString(PETSC_NULL, "-prefix", tmp_str, 256, &flag);
	if(flag)sprintf(prefix, "%s_", tmp_str);
	else sprintf(prefix, "");

        PetscOptionsGetInt(PETSC_NULL, "-LES", &LES, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, "-les_prt", &les_prt, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, "-temperature", &temperature, PETSC_NULL);

		
	PetscOptionsGetInt(PETSC_NULL, "-vc", &vc, PETSC_NULL);	
	PetscOptionsGetInt(PETSC_NULL, "-binary", &binary_input, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-xyz", &xyz_input, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-rans", &rans, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-ransout", &rans_output, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-levelset", &levelset, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-avg", &avg, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-shear", &shear, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-averaging", &averaging_option, &flag);	// from control.dat
	
	PetscOptionsGetInt(PETSC_NULL, "-cs", &cs, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-i_periodic", &i_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-j_periodic", &j_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-k_periodic", &k_periodic, &flag);
	
	PetscOptionsGetInt(PETSC_NULL, "-ii_periodic", &i_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-jj_periodic", &j_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-kk_periodic", &k_periodic, &flag);
	
	PetscOptionsGetInt(PETSC_NULL, "-nv", &nv_once, &flag);
	printf("nv_once=%d\n", nv_once);
	
	int QCR = 0;
	PetscOptionsGetInt(PETSC_NULL, "-qcr", &QCR, PETSC_NULL);
	
	PetscOptionsGetInt(PETSC_NULL, "-tis", &tis, &flag);
	if (!flag) PetscPrintf(PETSC_COMM_WORLD, "Need the starting number!\n");    
	
	PetscOptionsGetInt(PETSC_NULL, "-tie", &tie, &flag);
	if (!flag) tie = tis;
    
	PetscOptionsGetInt(PETSC_NULL, "-ts", &tsteps, &flag);
	if (!flag) tsteps = 5; /* Default increasement is 5 */
	PetscOptionsGetInt(PETSC_NULL, "-steps_avg", &steps_avg, &flag);
   
 
	PetscOptionsGetInt(PETSC_NULL, "-onlyV", &onlyV, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-iavg", &i_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-javg", &j_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-kavg", &k_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-ikavg", &ik_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-pcr", &pcr, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-reynolds", &reynolds, &flag);
	
	PetscOptionsGetInt(PETSC_NULL, "-ikcavg", &ikc_average, &flag);

	PetscOptionsGetInt(PETSC_NULL, "-rotor_model", &rotor_model, PETSC_NULL);

	PetscOptionsGetInt(PETSC_NULL, "-NumberOfTurbines", &NumberOfTurbines, PETSC_NULL);
	if(flag) {
		PetscTruth flag1, flag2;
		PetscOptionsGetInt(PETSC_NULL, "-pi", &pi, &flag1);
		PetscOptionsGetInt(PETSC_NULL, "-pk", &pk, &flag2);
		
		if(!flag1 || !flag2) {
			printf("To use -ikcavg you must set -pi and -pk, which are number of points in i- and k- directions.\n");
			exit(0);
		}

	}
	
/*  
	if(pcr) avg=1;
	if(i_average) avg=1;
	if(j_average) avg=1;
	if(k_average) avg=1;
	if(ik_average) avg=1;
	if(ikc_average) avg=1;
*/  
  
//	if(i_average + j_average + k_average >1) PetscPrintf(PETSC_COMM_WORLD, "Iavg and Javg cannot be set together !! !\n"), exit(0);
      
	PetscInt rank, bi;

	PetscMalloc(sizeof(IBMNodes), &ibm0);
	PetscMalloc(sizeof(IBMNodes), &ibm1);

	FSInfo *fsi_wt;

        if (rotor_model) {

	
		printf("the number of turbines %d \n", NumberOfTurbines);
		PetscMalloc(NumberOfTurbines*sizeof(IBMNodes), &wtm);
    		PetscMalloc(NumberOfTurbines*sizeof(FSInfo), &fsi_wt);

		int ibi;

                if (!rank) {

                        FILE *fd;
                        char str[256];
                        sprintf(str, "./CenterWT.dat");
                        fd = fopen(str, "r");
                        if(!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(0);
                        for (ibi=0;ibi<NumberOfTurbines;ibi++) {
                                fscanf(fd, "%le %le %le ", &(fsi_wt[ibi].x_c), &(fsi_wt[ibi].y_c), &(fsi_wt[ibi].z_c));
                                MPI_Bcast(&(fsi_wt[ibi].x_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                MPI_Bcast(&(fsi_wt[ibi].y_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                MPI_Bcast(&(fsi_wt[ibi].z_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                PetscPrintf(PETSC_COMM_WORLD, "The rotating center %f %f %f \n", (fsi_wt[ibi].x_c), (fsi_wt[ibi].y_c),
(fsi_wt[ibi].z_c));
                        }

                        fclose(fd);

                }
                else {

                        for (ibi=0;ibi<NumberOfTurbines;ibi++) {
                                MPI_Bcast(&(fsi_wt[ibi].x_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                MPI_Bcast(&(fsi_wt[ibi].y_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                MPI_Bcast(&(fsi_wt[ibi].z_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
//                              PetscPrintf(PETSC_COMM_WORLD, "The rotating center %f %f %f \n", (fsi_wt[ibi].x_c), &(fsi_wt[ibi].y_c),
//                              &(fsi_wt[ibi].z_c));
                        }



                }

	}


	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if(xyz_input) {block_number=1;}
	else {
		FILE *fd;
		fd = fopen("grid.dat", "r");
		if(binary_input) fread(&block_number, sizeof(int), 1, fd);
		else fscanf(fd, "%i\n", &block_number);
		MPI_Bcast(&block_number, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		fclose(fd);
	}
	
	PetscMalloc(block_number*sizeof(UserCtx), &user);
	PetscOptionsGetReal(PETSC_NULL, "-ren", &user->ren, PETSC_NULL);

	ReadCoordinates(user);

	PetscPrintf(PETSC_COMM_WORLD, "read coord!\n");


        for (bi=0; bi<block_number; bi++) {
        	DACreateGlobalVector(user[bi].fda, &user[bi].Csi);
               	DACreateGlobalVector(user[bi].fda, &user[bi].Eta);
               	DACreateGlobalVector(user[bi].fda, &user[bi].Zet);
            	DACreateGlobalVector(user[bi].da, &user[bi].Aj);

             	FormMetrics(&(user[bi]));
	}                

        if (rotor_model) {

		int i;
                for (i=0;i<NumberOfTurbines;i++) {

                        PetscPrintf(PETSC_COMM_WORLD, "Turbines read!\n");

                        if (rotor_model == 1) ACD_read(&wtm[i], i, &fsi_wt[i]);
                        // if (rotor_model == 2 || rotor_model == 3) ibm_read_ACL(&wtm[i], i);

                        // if (rotor_model == 3) ACD_read(&ibm_ACD[i], i);

//			Pre_process(&(user[0]), wtm);
                        PetscBarrier(PETSC_NULL);


			printf("here \n");
                }

		
		Pre_process(&(user[0]), wtm);

        }


	for (bi=0; bi<block_number; bi++) {
		DACreateGlobalVector(user[bi].da, &user[bi].Nvert);
		if(shear) {
			Calc_avg_shear_stress(&(user[bi]));
			exit(0);
		}
		else if(!avg) {
			DACreateGlobalVector(user[bi].da, &user[bi].P);
			DACreateGlobalVector(user[bi].fda, &user[bi].Ucat);

			if(!vc) DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_o);
			
		}
		else {
			if(pcr) {
				DACreateGlobalVector(user[bi].da, &user[bi].P);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat);
			}
			else if(avg==1) {
			  /*
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_cross_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);
			  */
			}
			else if(avg==2) {	// just compute k
			  /*
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);
			  */
			}
		}
		
	}


  
	if(avg) {
		if(i_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in I direction!\n");
		else if(j_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in J direction!\n");
		else if(k_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in K direction!\n");
		else if(ik_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in IK direction!\n");
		else PetscPrintf(PETSC_COMM_WORLD, "Averaging !\n");
		/*
		DACreateGlobalVector(user[bi].fda, &user->Ucat_sum);
		DACreateGlobalVector(user[bi].fda, &user->Ucat_cross_sum);
		DACreateGlobalVector(user[bi].fda, &user->Ucat_square_sum);
		*/
		
	}
 
        for (bi=0; bi<block_number; bi++) {

                VecDestroy(user[bi].Csi);
                VecDestroy(user[bi].Eta);
                VecDestroy(user[bi].Zet);
                VecDestroy(user[bi].Aj);
        }

 
	for (ti=tis; ti<=tie; ti+=tsteps) {
		for (bi=0; bi<block_number; bi++) {
			if(avg) Ucont_P_Binary_Input_Averaging(&user[bi]);
			else {
				Ucont_P_Binary_Input(&user[bi]);
				Ucont_P_Binary_Input1(&user[bi]);
			}
		}
	
	
		if (!QCR) {

			if(avg) TECIOOut_Averaging(user, wtm);
			else TECIOOut_V_2D(user, onlyV);
			//TECIOOut(user);
		}
		else {
			TECIOOutQ(user, QCR);
		}
	}
	PetscFinalize();
}



PetscErrorCode ReadCoordinates(UserCtx *user)
{
	Cmpnts ***coor;

	Vec Coor;
	PetscInt bi, i, j, k, rank, IM, JM, KM;
	PetscReal *gc;
	FILE *fd;
	PetscReal	d0 = 1.;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscReal	cl = 1.;
	PetscOptionsGetReal(PETSC_NULL, "-cl", &cl, PETSC_NULL);

	char str[256];
	
	if(xyz_input) sprintf(str, "xyz.dat");
	else sprintf(str, "grid.dat");
	
	fd = fopen(str, "r");
	
	if(fd==NULL) printf("Cannot open %s !\n", str),exit(0);

	printf("Begin Reading %s !\n", str);
	  
	if(xyz_input) {i=1;}
	else if(binary_input) {
		fread(&i, sizeof(int), 1, fd);
		if(i!=1) PetscPrintf(PETSC_COMM_WORLD, "This seems to be a text file !\n"),exit(0);
	}
	else {
		fscanf(fd, "%i\n", &i);
		if(i!=1) PetscPrintf(PETSC_COMM_WORLD, "This seems to be a binary file !\n"),exit(0);
	}
	

	for (bi=block_number-1; bi>=0; bi--) {
	  
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
		
		IM = user[bi].IM; JM = user[bi].JM; KM = user[bi].KM;
    
    
		DACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX,
				user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, 1,1,
				PETSC_DECIDE, 1, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL,
				&(user[bi].da));
		if(rans) {
			DACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX,
				user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, 1,1,
				PETSC_DECIDE, 2, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL,
				&(user[bi].fda2));
		}
		DASetUniformCoordinates(user[bi].da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
		DAGetCoordinateDA(user[bi].da, &(user[bi].fda));
	
		DAGetLocalInfo(user[bi].da, &(user[bi].info));

		DALocalInfo	info = user[bi].info;
		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;
		
		DAGetGhostedCoordinates(user[bi].da, &Coor);
		DAVecGetArray(user[bi].fda, Coor, &coor);
		
		double buffer;
		
		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if(xyz_input) {}
			else if(binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);
				
			if( k>=zs && k<=ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if(xyz_input) coor[k][j][i].x = X[i]/cl;
				else coor[k][j][i].x = buffer/cl;
			}
		}
			
		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if(xyz_input) {}
			else if(binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);
				
			if( k>=zs && k<=ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if(xyz_input) coor[k][j][i].y = Y[j]/cl;
				else coor[k][j][i].y = buffer/cl;
			}
		}
	
		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if(xyz_input) {}
			else if(binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);
				
			if( k>=zs && k<=ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if(xyz_input) coor[k][j][i].z = Z[k]/cl;
				else coor[k][j][i].z = buffer/cl;
			}
		}

		DAVecRestoreArray(user[bi].fda, Coor, &coor);

		Vec	gCoor;
		DAGetCoordinates(user[bi].da, &gCoor);
		DALocalToGlobal(user[bi].fda, Coor, INSERT_VALUES, gCoor);
		DAGlobalToLocalBegin(user[bi].fda, gCoor, INSERT_VALUES, Coor);
		DAGlobalToLocalEnd(user[bi].fda, gCoor, INSERT_VALUES, Coor);

	}
	
	fclose(fd);
	
	printf("Finish Reading %s !\n", str);
  
	for (bi=0; bi<block_number; bi++) {
		user[bi]._this = bi;
	}
	return(0);
}

void Calc_avg_shear_stress(UserCtx *user)
{
	double N=(double)tis+1.0;
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;

  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  PetscInt i, j, k;
  lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

  Cmpnts ***usum, ***csi, ***eta, ***zet;
  PetscReal ***aj, ***psum, ***nvert;
  if (lxs==0) lxs++;
  if (lxe==mx) lxe--;
  if (lys==0) lys++;
  if (lye==my) lye--;
  if (lzs==0) lzs++;
  if (lze==mz) lze--;

	char filen[256];
	PetscViewer	viewer;
		
	Vec P_sum;
	DACreateGlobalVector(user->da, &P_sum);
  DACreateGlobalVector(user->fda, &user->Ucat_sum);
  	  
  ti=tis;
  sprintf(filen, "su0_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_sum));
	PetscViewerDestroy(viewer);
	
	ti=tis;		
	sprintf(filen, "sp_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, P_sum);
	PetscViewerDestroy(viewer);

  DAVecGetArray(user->fda, user->Csi, &csi);
  DAVecGetArray(user->fda, user->Eta, &eta);
  DAVecGetArray(user->fda, user->Zet, &zet);
  DAVecGetArray(user->da, user->Aj, &aj);
  DAVecGetArray(user->da, user->Nvert, &nvert);
  DAVecGetArray(user->fda, user->Ucat_sum, &usum);
  DAVecGetArray(user->da, P_sum, &psum);


	double force_skin_bottom = 0;
	double force_pressure_bottom = 0;
	double force_bottom = 0;
	double area_bottom = 0;
	
	double force_skin_top = 0;
	double force_pressure_top = 0;
	double force_top = 0;
	double area_top = 0;
			
	j=0;
	for (k=lzs; k<lze; k++)
	for (i=lxs; i<lxe; i++) {
		if (nvert[k][j+1][i] < 0.1) {
			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
			double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
			
			dudc=0, dvdc=0, dwdc=0;
			
			dude=usum[k][j+1][i].x * 2.0 / N;
			dvde=usum[k][j+1][i].y * 2.0 / N;
			dwde=usum[k][j+1][i].z * 2.0 / N;
			
			dudz=0, dvdz=0, dwdz=0;
			
			double ajc = aj[k][j+1][i];
			double csi0 = csi[k][j+1][i].x, csi1 = csi[k][j+1][i].y, csi2 = csi[k][j+1][i].z;
			double eta0 = eta[k][j+1][i].x, eta1 = eta[k][j+1][i].y, eta2 = eta[k][j+1][i].z;
			double zet0 = zet[k][j+1][i].x, zet1 = zet[k][j+1][i].y, zet2 = zet[k][j+1][i].z;

			Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, 
					dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

			double j_area = sqrt( eta[k][j+1][i].x*eta[k][j+1][i].x + eta[k][j+1][i].y*eta[k][j+1][i].y + eta[k][j+1][i].z*eta[k][j+1][i].z );
			double ni[3], nj[3], nk[3];
			double nx, ny, nz;
			Calculate_normal(csi[k][j+1][i], eta[k][j+1][i], zet[k][j+1][i], ni, nj, nk);
			nx = nj[0]; //inward normal
			ny = nj[1]; //inward normal
			nz = nj[2]; //inward normal
			
			
			double Fp = - psum[k][j+1][i] * eta2 / N;
			double Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
			//double Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * j_area;
			
			force_skin_bottom += Fs;
			force_pressure_bottom += Fp;
			force_bottom += Fs + Fp;
			area_bottom += fabs(eta1);	// projected area
		}
	}

	j=my-2;
	for (k=lzs; k<lze; k++)
	for (i=lxs; i<lxe; i++) {
		if (nvert[k][j][i] < 0.1) {
			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
			double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
			
			dudc=0, dvdc=0, dwdc=0;
			
			dude = -usum[k][j][i].x * 2.0 / N;
			dvde = -usum[k][j][i].y * 2.0 / N;
			dwde = -usum[k][j][i].z * 2.0 / N;
			
			dudz=0, dvdz=0, dwdz=0;
			
			double ajc = aj[k][j][i];
			double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
			double eta0 = eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
			double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;

			Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, 
					dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

			double j_area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
			double ni[3], nj[3], nk[3];
			double nx, ny, nz;
			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
			nx = -nj[0]; //inward normal
			ny = -nj[1]; //inward normal
			nz = -nj[2]; //inward normal
			
			
			double Fp = - psum[k][j][i] * eta2 / N;
			double Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
			//double Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * j_area;
			
			force_skin_top += Fs;
			force_pressure_top += Fp;
			force_top += Fs + Fp;
			area_top += fabs(eta1);	// projected area
		}
	}
	
  DAVecRestoreArray(user->fda, user->Csi, &csi);
  DAVecRestoreArray(user->fda, user->Eta, &eta);
  DAVecRestoreArray(user->fda, user->Zet, &zet);
  DAVecRestoreArray(user->da, user->Aj, &aj);
  DAVecRestoreArray(user->da, user->Nvert, &nvert);
  DAVecRestoreArray(user->fda, user->Ucat_sum, &usum);
  DAVecRestoreArray(user->da, P_sum, &psum);

	VecDestroy(P_sum);
  VecDestroy(user->Ucat_sum);
  
  printf("Top:\tarea=%f, force=%f, skin force=%f, pressure force=%f\n",
  			area_top, force_top, force_skin_top, force_pressure_top);
  			
	printf("\tstress=%f, skin stress=%f, pressure stress=%f, u*=%f, Re*=%f\n",
  			force_top/area_top, force_skin_top/area_top, force_pressure_top/area_top,
  			sqrt(fabs(force_top/area_top)), sqrt(fabs(force_top/area_top))*user->ren);
  			
  printf("\n");
  
  printf("Bottom:\tarea=%f, force=%f, skin force=%f, pressure force=%f\n",
  			area_bottom, force_bottom, force_skin_bottom, force_pressure_bottom);
  			
	printf("\tstress=%f, skin stress=%f, pressure stress=%f, u*=%f, Re*=%f\n",
  			force_bottom/area_bottom, force_skin_bottom/area_bottom, force_pressure_bottom/area_bottom,
  			sqrt(fabs(force_bottom/area_bottom)), sqrt(fabs(force_bottom/area_bottom))*user->ren);
}

PetscErrorCode Lambda2(UserCtx *user)
{
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;

  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  PetscInt i, j, k;
  lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

  Cmpnts ***ucat, ***csi, ***eta, ***zet;
  PetscReal ***aj, ***q, ***nvert;
  if (lxs==0) lxs++;
  if (lxe==mx) lxe--;
  if (lys==0) lys++;
  if (lye==my) lye--;
  if (lzs==0) lzs++;
  if (lze==mz) lze--;

  DAVecGetArray(user->fda, user->Ucat, &ucat);
  DAVecGetArray(user->fda, user->Csi, &csi);
  DAVecGetArray(user->fda, user->Eta, &eta);
  DAVecGetArray(user->fda, user->Zet, &zet);

  DAVecGetArray(user->da, user->Aj, &aj);
  DAVecGetArray(user->da, user->Nvert, &nvert);
  DAVecGetArray(user->da, user->P, &q);

  PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;

  PetscReal s11, s12, s13, s21, s22, s23, s31, s32, s33;
  PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

  PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
  //PetscReal so, wo;
  PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	      
	double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
	double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
	double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
	double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
	double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
	double ajc = aj[k][j][i];
	
	Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
	Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
							
	double Sxx = 0.5*( du_dx + du_dx ), Sxy = 0.5*(du_dy + dv_dx), Sxz = 0.5*(du_dz + dw_dx);
	double Syx = Sxy, Syy = 0.5*(dv_dy + dv_dy),	Syz = 0.5*(dv_dz + dw_dy);
	double Szx = Sxz, Szy=Syz, Szz = 0.5*(dw_dz + dw_dz);

	
	w11 = 0;
	w12 = 0.5*(du_dy - dv_dx);
	w13 = 0.5*(du_dz - dw_dx);
	w21 = -w12;
	w22 = 0.;
	w23 = 0.5*(dv_dz - dw_dy);
	w31 = -w13;
	w32 = -w23;
	w33 = 0.;
	
	
	double S[3][3], W[3][3], D[3][3];
	
	D[0][0] = du_dx, D[0][1] = du_dy, D[0][2] = du_dz;
	D[1][0] = dv_dx, D[1][1] = dv_dy, D[1][2] = dv_dz;
	D[2][0] = dw_dx, D[2][1] = dw_dy, D[2][2] = dw_dz;
	
	S[0][0] = Sxx;
	S[0][1] = Sxy;
	S[0][2] = Sxz;

	S[1][0] = Syx;
	S[1][1] = Syy;
	S[1][2] = Syz;

	S[2][0] = Szx;
	S[2][1] = Szy;
	S[2][2] = Szz;

	W[0][0] = w11;
	W[0][1] = w12;
	W[0][2] = w13;
	W[1][0] = w21;
	W[1][1] = w22;
	W[1][2] = w23;
	W[2][0] = w31;
	W[2][1] = w32;
	W[2][2] = w33;
	
	// lambda-2
	double A[3][3], V[3][3], d[3];
	
	for(int row=0; row<3; row++)
	for(int col=0; col<3; col++) A[row][col]=0;
	
	for(int row=0; row<3; row++)
	for(int col=0; col<3; col++) {
		A[row][col] += S[row][0] * S[0][col];
		A[row][col] += S[row][1] * S[1][col];
		A[row][col] += S[row][2] * S[2][col];
	}
	
	for(int row=0; row<3; row++)
	for(int col=0; col<3; col++) {
		A[row][col] += W[row][0] * W[0][col];
		A[row][col] += W[row][1] * W[1][col];
		A[row][col] += W[row][2] * W[2][col];
	}
	
	if(nvert[k][j][i]<0.1) {
		eigen_decomposition(A, V, d);
		q[k][j][i] = d[1];
	}
	else q[k][j][i] = 1000.0;
/*	
	// delta criterion
	double DD[3][3];
	for(int row=0; row<3; row++)
	for(int col=0; col<3; col++) DD[row][col]=0;
	
	for(int row=0; row<3; row++)
	for(int col=0; col<3; col++) {
		DD[row][col] += D[row][0] * D[0][col];
		DD[row][col] += D[row][1] * D[1][col];
		DD[row][col] += D[row][2] * D[2][col];
	}
	double tr_DD = DD[0][0] + DD[1][1] + DD[2][2];
	double det_D = D[0][0]*(D[2][2]*D[1][1]-D[2][1]*D[1][2])-D[1][0]*(D[2][2]*D[0][1]-D[2][1]*D[0][2])+D[2][0]*(D[1][2]*D[0][1]-D[1][1]*D[0][2]);
	
	//double Q = -0.5*tr_DD;
	
	double SS=0, WW=0;
	for(int row=0; row<3; row++)
	for(int col=0; col<3; col++) {
		SS+=S[row][col]*S[row][col];
		WW+=W[row][col]*W[row][col];
	}
	double Q = 0.5*(WW - SS);
	
	double R = - det_D;
	if(nvert[k][j][i]<0.1) {
		q[k][j][i] = pow( 0.5*R, 2. )  + pow( Q/3., 3.);
	}
	else q[k][j][i] = -10;
	if(q[k][j][i]<0) q[k][j][i]=-10;
*/	
      }
    }
  }

  DAVecRestoreArray(user->fda, user->Ucat, &ucat);
  DAVecRestoreArray(user->fda, user->Csi, &csi);
  DAVecRestoreArray(user->fda, user->Eta, &eta);
  DAVecRestoreArray(user->fda, user->Zet, &zet);

  DAVecRestoreArray(user->da, user->Aj, &aj);
  DAVecRestoreArray(user->da, user->Nvert, &nvert);
  DAVecRestoreArray(user->da, user->P, &q);

  return 0;
}

PetscErrorCode QCriteria(UserCtx *user)
{

  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;

  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  PetscInt i, j, k;
  lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

  Cmpnts ***ucat, ***csi, ***eta, ***zet;
  PetscReal ***aj, ***q, ***nvert;
  if (lxs==0) lxs++;
  if (lxe==mx) lxe--;
  if (lys==0) lys++;
  if (lye==my) lye--;
  if (lzs==0) lzs++;
  if (lze==mz) lze--;
	
  DAVecGetArray(user->fda, user->Ucat, &ucat);
  DAVecGetArray(user->fda, user->Csi, &csi);
  DAVecGetArray(user->fda, user->Eta, &eta);
  DAVecGetArray(user->fda, user->Zet, &zet);

  DAVecGetArray(user->da, user->Aj, &aj);
  DAVecGetArray(user->da, user->Nvert, &nvert);
  DAVecGetArray(user->da, user->P, &q);

  PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;

  PetscReal s11, s12, s13, s21, s22, s23, s31, s32, s33;
  PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

  PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
  PetscReal so, wo;
  PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;
  
  
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
	double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
	double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
	double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
	double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
	double ajc = aj[k][j][i];
	
	Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
	Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
							&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
							
	double Sxx = 0.5*( du_dx + du_dx ), Sxy = 0.5*(du_dy + dv_dx), Sxz = 0.5*(du_dz + dw_dx);
	double Syx = Sxy, Syy = 0.5*(dv_dy + dv_dy),	Syz = 0.5*(dv_dz + dw_dy);
	double Szx = Sxz, Szy=Syz, Szz = 0.5*(dw_dz + dw_dz);
	so = Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + Syx*Syx + Syy*Syy + Syz*Syz + Szx*Szx + Szy*Szy + Szz*Szz;
	
	w11 = 0;
	w12 = 0.5*(du_dy - dv_dx);
	w13 = 0.5*(du_dz - dw_dx);
	w21 = -w12;
	w22 = 0.;
	w23 = 0.5*(dv_dz - dw_dy);
	w31 = -w13;
	w32 = -w23;
	w33 = 0.;
	
	wo = w11*w11 + w12*w12 + w13*w13 + w21*w21 + w22*w22 + w23*w23 + w31*w31 + w32*w32 + w33*w33;

/*
	so = ( d11 *  d11 + d22 * d22 + d33 * d33) + 0.5* ( (d12 + d21) * (d12 + d21) + (d13 + d31) * (d13 + d31) + (d23 + d32) * (d23 + d32) );
	wo = 0.5 * ( (d12 - d21)*(d12 - d21) + (d13 - d31) * (d13 - d31) + (d23 - d32) * (d23 - d32) );
	V19=0.5 * ( (V13 - V11)*(V13 - V11) + (V16 - V12) * (V16 - V12) + (V17 - V15) * (V17 - V15) ) - 0.5 * ( V10 *  V10 + V14 * V14 + V18 * V18) - 0.25* ( (V13 + V11) * (V13 + V11) + (V16 + V12) * (V16 + V12) + (V17 + V15) * (V17 + V15) )
*/
	
	if( nvert[k][j][i]>0.1 ) q[k][j][i] = -100;
	else q[k][j][i] = (wo - so) / 2.;
      }
    }
  }

	DAVecRestoreArray(user->fda, user->Ucat, &ucat);
	DAVecRestoreArray(user->fda, user->Csi, &csi);
	DAVecRestoreArray(user->fda, user->Eta, &eta);
	DAVecRestoreArray(user->fda, user->Zet, &zet);

	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->Nvert, &nvert);
	DAVecRestoreArray(user->da, user->P, &q);

  return 0;
}

PetscErrorCode Velocity_Magnitude(UserCtx *user)	// store at P
{
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;

  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  PetscInt i, j, k;
  lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

  Cmpnts ***ucat;
  PetscReal ***aj, ***q, ***nvert;
  if (lxs==0) lxs++;
  if (lxe==mx) lxe--;
  if (lys==0) lys++;
  if (lye==my) lye--;
  if (lzs==0) lzs++;
  if (lze==mz) lze--;

  DAVecGetArray(user->fda, user->Ucat, &ucat);
  DAVecGetArray(user->da, user->P, &q);

  PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;
  PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	q[k][j][i] = sqrt( ucat[k][j][i].x*ucat[k][j][i].x + ucat[k][j][i].y*ucat[k][j][i].y + ucat[k][j][i].z*ucat[k][j][i].z );
      }
    }
  }

  DAVecRestoreArray(user->fda, user->Ucat, &ucat);
  DAVecRestoreArray(user->da, user->P, &q);

  return 0;
}


PetscErrorCode ibm_read(IBMNodes *ibm)
{
  PetscInt	rank;
  PetscInt	n_v , n_elmt ;
  PetscReal	*x_bp , *y_bp , *z_bp ;
  PetscInt	*nv1 , *nv2 , *nv3 ;
  PetscReal	*nf_x, *nf_y, *nf_z;
  PetscInt	i;
  PetscInt	n1e, n2e, n3e;
  PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  PetscReal	t, dr;
  double xt;
  //MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if(!rank) { // root processor read in the data
    FILE *fd;
    fd = fopen("ibmdata0", "r"); if (!fd) SETERRQ(1, "Cannot open IBM node file")
    n_v =0;
    fscanf(fd, "%i", &n_v);
    fscanf(fd, "%i", &n_v);
    fscanf(fd, "%le", &xt);
    ibm->n_v = n_v;

    MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    //    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));

    for (i=0; i<n_v; i++) {
      fscanf(fd, "%le %le %le %le %le %le", &x_bp[i], &y_bp[i], &z_bp[i], &t, &t, &t);
      x_bp[i] = x_bp[i] / 28.;
      y_bp[i] = y_bp[i] / 28.;
      z_bp[i] = z_bp[i] / 28.;
    }
    ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp;

    for (i=0; i<n_v; i++) {
      PetscReal temp;
      temp = ibm->y_bp0[i];
      ibm->y_bp0[i] = ibm->z_bp0[i];
      ibm->z_bp0[i] = -temp;
    }


    MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    fscanf(fd, "%i\n", &n_elmt);
    ibm->n_elmt = n_elmt;
    MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

    PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
    for (i=0; i<n_elmt; i++) {

      fscanf(fd, "%i %i %i\n", nv1+i, nv2+i, nv3+i);
      nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
      //      PetscPrintf(PETSC_COMM_WORLD, "I %d %d %d\n", nv1[i], nv2[i], nv3[i]);
    }
    ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

    fclose(fd);

    for (i=0; i<n_elmt; i++) {
      n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
      dx12 = x_bp[n2e] - x_bp[n1e];
      dy12 = y_bp[n2e] - y_bp[n1e];
      dz12 = z_bp[n2e] - z_bp[n1e];

      dx13 = x_bp[n3e] - x_bp[n1e];
      dy13 = y_bp[n3e] - y_bp[n1e];
      dz13 = z_bp[n3e] - z_bp[n1e];

      nf_x[i] = dy12 * dz13 - dz12 * dy13;
      nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      nf_z[i] = dx12 * dy13 - dy12 * dx13;

      dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);

      nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;

      
    }
    
    ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;

/*     for (i=0; i<n_elmt; i++) { */
/*       PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d\n", i, nv1[i], nv2[i], nv3[i]); */
/*     } */
    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;

    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
/*     ibm->x_bp0 = x_bp;  ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */

    MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_elmt = n_elmt;

    PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;

    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

/*     MPI_Bcast(&(ibm->nv1), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv2), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv3), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */

/*     MPI_Bcast(&(ibm->nf_x), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_y), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_z), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
  }

/*   MPI_Barrier(PETSC_COMM_WORLD); */
  return(0);
}

PetscErrorCode ibm_read_ucd(IBMNodes *ibm)
{
  PetscInt	rank;
  PetscInt	n_v , n_elmt ;
  PetscReal	*x_bp , *y_bp , *z_bp ;
  PetscInt	*nv1 , *nv2 , *nv3 ;
  PetscReal	*nf_x, *nf_y, *nf_z;
  PetscInt	i;
  PetscInt	n1e, n2e, n3e;
  PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  PetscReal	t, dr;
  PetscInt 	temp;
  double xt;
  char string[128];
  //MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if(!rank) { // root processor read in the data
    FILE *fd;
    fd = fopen("ibmdata", "r");
    if (!fd) SETERRQ(1, "Cannot open IBM node file")
    n_v =0;

    if (fd) {
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      fgets(string, 128, fd);

      fscanf(fd, "%i %i %i %i %i\n", &n_v, &n_elmt, &temp, &temp, &temp);
      
      ibm->n_v = n_v;
      ibm->n_elmt = n_elmt;

      MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      //    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
      PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));

      
      PetscReal cl = 1.;

      PetscOptionsGetReal(PETSC_NULL, "-chact_leng_valve", &cl, PETSC_NULL);
      
      for (i=0; i<n_v; i++) {
	fscanf(fd, "%i %le %le %le", &temp, &x_bp[i], &y_bp[i], &z_bp[i]);
	x_bp[i] = x_bp[i] / cl;
	y_bp[i] = y_bp[i] / cl;
	z_bp[i] = z_bp[i] / cl;
	
	ibm->x_bp[i] = x_bp[i];
	ibm->y_bp[i] = y_bp[i];
	ibm->z_bp[i] = z_bp[i];

	ibm->u[i].x = 0.;
	ibm->u[i].y = 0.;
	ibm->u[i].z = 0.;
      }
      ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp;

      MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

	

      MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
      PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
      PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      char str[20];
      for (i=0; i<n_elmt; i++) {

	fscanf(fd, "%i %i %s %i %i %i\n", &temp, &temp, str, nv1+i, nv2+i, nv3+i);
	nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;

      }
      ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

      fclose(fd);
    }
    for (i=0; i<n_elmt; i++) {
      n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
      dx12 = x_bp[n2e] - x_bp[n1e];
      dy12 = y_bp[n2e] - y_bp[n1e];
      dz12 = z_bp[n2e] - z_bp[n1e];

      dx13 = x_bp[n3e] - x_bp[n1e];
      dy13 = y_bp[n3e] - y_bp[n1e];
      dz13 = z_bp[n3e] - z_bp[n1e];

      nf_x[i] = dy12 * dz13 - dz12 * dy13;
      nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      nf_z[i] = dx12 * dy13 - dy12 * dx13;

      dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);

      nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;

      
    }
    
    ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;

    /*     for (i=0; i<n_elmt; i++) { */
    /*       PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d\n", i, nv1[i], nv2[i], nv3[i]); */
    /*     } */
    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;

    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));

    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));

/*     ibm->x_bp0 = x_bp;  ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */

    MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    for (i=0; i<ibm->n_v; i++) {
      ibm->x_bp[i] = ibm->x_bp0[i];
      ibm->y_bp[i] = ibm->y_bp0[i];
      ibm->z_bp[i] = ibm->z_bp0[i];

      ibm->u[i].x = 0.;
      ibm->u[i].y = 0.;
      ibm->u[i].z = 0.;
    }
    MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_elmt = n_elmt;

    PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;

    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

/*     MPI_Bcast(&(ibm->nv1), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv2), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv3), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */

/*     MPI_Bcast(&(ibm->nf_x), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_y), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_z), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
  }

/*   MPI_Barrier(PETSC_COMM_WORLD); */
  return(0);
}

PetscErrorCode Combine_Elmt(IBMNodes *ibm, IBMNodes *ibm0, IBMNodes *ibm1)
{

  PetscInt i;

  ibm->n_v = ibm0->n_v + ibm1->n_v;
  ibm->n_elmt = ibm0->n_elmt + ibm1->n_elmt;

  PetscInt n_v = ibm->n_v, n_elmt = ibm->n_elmt;

  for (i=0; i<ibm0->n_v; i++) {
    ibm->x_bp[i] = ibm0->x_bp[i];
    ibm->y_bp[i] = ibm0->y_bp[i];
    ibm->z_bp[i] = ibm0->z_bp[i];

    ibm->u[i] = ibm0->u[i];
    ibm->uold[i] = ibm0->uold[i];
    //    ibm->u[i].x = 0.;
/*     PetscPrintf(PETSC_COMM_WORLD, "Vel %e %e %e\n", ibm->u[i].x, ibm->u[i].y, ibm->u[i].z); */
  }
  for (i=0; i<ibm0->n_elmt; i++) {
    ibm->nv1[i] = ibm0->nv1[i];
    ibm->nv2[i] = ibm0->nv2[i];
    ibm->nv3[i] = ibm0->nv3[i];

    ibm->nf_x[i] = ibm0->nf_x[i];
    ibm->nf_y[i] = ibm0->nf_y[i];
    ibm->nf_z[i] = ibm0->nf_z[i];
  }

  for (i=ibm0->n_v; i<n_v; i++) {
    ibm->x_bp[i] = ibm1->x_bp[i-ibm0->n_v];
    ibm->y_bp[i] = ibm1->y_bp[i-ibm0->n_v];
    ibm->z_bp[i] = ibm1->z_bp[i-ibm0->n_v];
    ibm->u[i].x = 0.;
    ibm->u[i].y = 0.;
    ibm->u[i].z = 0.;
  }

  for (i=ibm0->n_elmt; i<n_elmt; i++) {
    ibm->nv1[i] = ibm1->nv1[i-ibm0->n_elmt] + ibm0->n_v;
    ibm->nv2[i] = ibm1->nv2[i-ibm0->n_elmt] + ibm0->n_v;
    ibm->nv3[i] = ibm1->nv3[i-ibm0->n_elmt] + ibm0->n_v;

    ibm->nf_x[i] = ibm1->nf_x[i-ibm0->n_elmt];
    ibm->nf_y[i] = ibm1->nf_y[i-ibm0->n_elmt];
    ibm->nf_z[i] = ibm1->nf_z[i-ibm0->n_elmt];
  }

  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (!rank) {
    if (ti == ti) {
      FILE *f;
      char filen[80];
      sprintf(filen, "surface%06d.dat",ti);
      f = fopen(filen, "w");
      PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n");
      PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, 1670-96);
      for (i=0; i<n_v; i++) {

	/*    ibm->x_bp[i] = ibm->x_bp0[i];
	      ibm->y_bp[i] = ibm->y_bp0[i];
	      ibm->z_bp[i] = ibm->z_bp0[i] + z0;*/
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
      }
      for (i=96; i<n_elmt; i++) {
	if (fabs(ibm->nf_z[i]) > 0.5 ||
	    (fabs(ibm->nf_z[i]) < 0.5 &&
	     (ibm->x_bp[ibm->nv1[i]] * ibm->x_bp[ibm->nv1[i]] +
	      ibm->y_bp[ibm->nv1[i]] * ibm->y_bp[ibm->nv1[i]]) < 0.44*0.44)) {
	  PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
	}
      }
      fclose(f);

      sprintf(filen, "leaflet%06d.dat",ti);
      f = fopen(filen, "w");
      PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n");
      PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, 96);
      for (i=0; i<n_v; i++) {

	/*    ibm->x_bp[i] = ibm->x_bp0[i];
	      ibm->y_bp[i] = ibm->y_bp0[i];
	      ibm->z_bp[i] = ibm->z_bp0[i] + z0;*/
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
      }
      for (i=0; i<96; i++) {
	if (fabs(ibm->nf_z[i]) > 0.5 ||
	    (fabs(ibm->nf_z[i]) < 0.5 &&
	     (ibm->x_bp[ibm->nv1[i]] * ibm->x_bp[ibm->nv1[i]] +
	      ibm->y_bp[ibm->nv1[i]] * ibm->y_bp[ibm->nv1[i]]) < 0.44*0.44)) {
	  PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
	}
      }
      fclose(f);

    }
  }

  return 0;
}

PetscErrorCode Elmt_Move(IBMNodes *ibm, UserCtx *user)
{
  PetscInt n_v = ibm->n_v, n_elmt = ibm->n_elmt;

  PetscReal rcx = -0.122, rcz = -0.32, z0 = 4.52;
  rcx = -0.09450115; rcz = -0.3141615; z0 = 4.47;
  PetscReal dz;
  dz = -0.031;
  rcz = rcz-dz;
  rcx = rcx - dz * sin(10./180.*3.1415926);
  PetscReal temp;
  PetscInt i;

  PetscInt n1e, n2e, n3e;
  PetscReal dx12, dy12, dz12, dx13, dy13, dz13, dr;
  for (i=0; i<n_v; i++) {
    ibm->x_bp_o[i] = ibm->x_bp[i];
    ibm->y_bp_o[i] = ibm->y_bp[i];
    ibm->z_bp_o[i] = ibm->z_bp[i];
  }

  angle =-angle * 3.1415926/180.;
  //angle = 0;
  for (i=0; i<n_v/2; i++) {
    ibm->x_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * cos(angle) - (ibm->z_bp0[i] - rcz) * sin(angle) + rcx;
    ibm->y_bp[i] = ibm->y_bp0[i];
    ibm->z_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * sin(angle) + (ibm->z_bp0[i] - rcz) * cos(angle) + z0 + rcz;

  }
  rcx = -rcx;
  for (i=n_v/2; i<n_v; i++) {
    ibm->x_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * cos(-angle) - (ibm->z_bp0[i] - rcz) * sin(-angle) + rcx;
    ibm->y_bp[i] = ibm->y_bp0[i];
    ibm->z_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * sin(-angle) + (ibm->z_bp0[i] - rcz) * cos(-angle) + z0 + rcz;
  }

  /* Rotate 90 degree */
  for (i=0; i<n_v; i++) {
    temp = ibm->y_bp[i];
    ibm->y_bp[i] = ibm->x_bp[i];
    ibm->x_bp[i] = temp;
  }
  for (i=0; i<n_elmt; i++) {
      n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];
      dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
      dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
      dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];

      dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e];
      dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e];
      dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e];

      ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
      ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;

      dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i] +
		ibm->nf_z[i]*ibm->nf_z[i]);

      ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;

/*       PetscPrintf(PETSC_COMM_WORLD, "NFZ %d %d %d %d %e\n", i, ibm->nv1[i], ibm->nv2[i], ibm->nv3[i], ibm->nf_z[i]); */
      //      PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", x_bp[n1e], y_bp[n1e], ibm->x_bp0[n1e], ibm->y_bp0[n1e], x_bp[n3e], y_bp[n3e]);

  }
  if (ti>0) {
    for (i=0; i<n_v; i++) {
      //      ibm->uold[i] = ibm->u[i];

      ibm->u[i].x = (ibm->x_bp[i] - ibm->x_bp_o[i]) / user->dt;
      ibm->u[i].y = (ibm->y_bp[i] - ibm->y_bp_o[i]) / user->dt;
      ibm->u[i].z = (ibm->z_bp[i] - ibm->z_bp_o[i]) / user->dt;
    }
  }
  else {
    for (i=0; i<n_v; i++) {
      ibm->u[i].x = 0.;
      ibm->u[i].y = 0.;
      ibm->u[i].z = 0.;
    }
  }
  return 0;
}

PetscErrorCode Elmt_Move1(IBMNodes *ibm, UserCtx *user)
{
  PetscInt n_v = ibm->n_v, n_elmt = ibm->n_elmt;

  PetscReal rcx = -0.122, rcz = -0.32, z0 = 4.52;
  rcx = -0.09450115; rcz = -0.3141615; z0 = 4.47;
  PetscReal dz;
  dz = -0.031;
  rcz = rcz-dz;
  rcx = rcx - dz * sin(10./180.*3.1415926);
  PetscReal temp;
  PetscInt i;

  PetscInt n1e, n2e, n3e;
  PetscReal dx12, dy12, dz12, dx13, dy13, dz13, dr;
  for (i=0; i<n_v; i++) {
    ibm->x_bp_o[i] = ibm->x_bp[i];
    ibm->y_bp_o[i] = ibm->y_bp[i];
    ibm->z_bp_o[i] = ibm->z_bp[i];
  }

  angle =-angle * 3.1415926/180.;
  //angle = 0;
  for (i=0; i<n_v/2; i++) {
    ibm->x_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * cos(angle) - (ibm->z_bp0[i] - rcz) * sin(angle) + rcx;
    ibm->y_bp[i] = ibm->y_bp0[i];
    ibm->z_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * sin(angle) + (ibm->z_bp0[i] - rcz) * cos(angle) + z0 + rcz;

  }
  rcx = -rcx;
  for (i=n_v/2; i<n_v; i++) {
    ibm->x_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * cos(-angle) - (ibm->z_bp0[i] - rcz) * sin(-angle) + rcx;
    ibm->y_bp[i] = ibm->y_bp0[i];
    ibm->z_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * sin(-angle) + (ibm->z_bp0[i] - rcz) * cos(-angle) + z0 + rcz;
  }

  /* Rotate 90 degree */
  for (i=0; i<n_v; i++) {
    temp = ibm->y_bp[i];
    ibm->y_bp[i] = ibm->x_bp[i];
    ibm->x_bp[i] = temp;
  }
  for (i=0; i<n_elmt; i++) {
      n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];
      dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
      dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
      dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];

      dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e];
      dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e];
      dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e];

      ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
      ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;

      dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i] +
		ibm->nf_z[i]*ibm->nf_z[i]);

      ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;

/*       PetscPrintf(PETSC_COMM_WORLD, "NFZ %d %d %d %d %e\n", i, ibm->nv1[i], ibm->nv2[i], ibm->nv3[i], ibm->nf_z[i]); */
      //      PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", x_bp[n1e], y_bp[n1e], ibm->x_bp0[n1e], ibm->y_bp0[n1e], x_bp[n3e], y_bp[n3e]);

  }
  if (ti>0) {
    for (i=0; i<n_v; i++) {
      ibm->uold[i] = ibm->u[i];

      ibm->u[i].x = (ibm->x_bp[i] - ibm->x_bp_o[i]) / user->dt;
      ibm->u[i].y = (ibm->y_bp[i] - ibm->y_bp_o[i]) / user->dt;
      ibm->u[i].z = (ibm->z_bp[i] - ibm->z_bp_o[i]) / user->dt;
    }
  }
  else {
    for (i=0; i<n_v; i++) {
      ibm->u[i].x = 0.;
      ibm->u[i].y = 0.;
      ibm->u[i].z = 0.;
    }
  }
  return 0;
}


/*****************************************************************/
#ifdef MAX
#undef MAX
#endif

#define MAX(a, b) ((a)>(b)?(a):(b))

#define n 3

static double hypot2(double x, double y) {
  return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

static void tred2(double V[n][n], double d[n], double e[n]) {

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int j = 0; j < n; j++) {
    d[j] = V[n-1][j];
  }

  // Householder reduction to tridiagonal form.

  for (int i = n-1; i > 0; i--) {

    // Scale to avoid under/overflow.

    double scale = 0.0;
    double h = 0.0;
    for (int k = 0; k < i; k++) {
      scale = scale + fabs(d[k]);
    }
    if (scale == 0.0) {
      e[i] = d[i-1];
      for (int j = 0; j < i; j++) {
        d[j] = V[i-1][j];
        V[i][j] = 0.0;
        V[j][i] = 0.0;
      }
    } else {

      // Generate Householder vector.

      for (int k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      double f = d[i-1];
      double g = sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h = h - f * g;
      d[i-1] = f - g;
      for (int j = 0; j < i; j++) {
        e[j] = 0.0;
      }

      // Apply similarity transformation to remaining columns.

      for (int j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (int k = j+1; k <= i-1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (int j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      double hh = f / (h + h);
      for (int j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (int j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i-1; k++) {
          V[k][j] -= (f * e[k] + g * d[k]);
        }
        d[j] = V[i-1][j];
        V[i][j] = 0.0;
      }
    }
    d[i] = h;
  }

  // Accumulate transformations.

  for (int i = 0; i < n-1; i++) {
    V[n-1][i] = V[i][i];
    V[i][i] = 1.0;
    double h = d[i+1];
    if (h != 0.0) {
      for (int k = 0; k <= i; k++) {
        d[k] = V[k][i+1] / h;
      }
      for (int j = 0; j <= i; j++) {
        double g = 0.0;
        for (int k = 0; k <= i; k++) {
          g += V[k][i+1] * V[k][j];
        }
        for (int k = 0; k <= i; k++) {
          V[k][j] -= g * d[k];
        }
      }
    }
    for (int k = 0; k <= i; k++) {
      V[k][i+1] = 0.0;
    }
  }
  for (int j = 0; j < n; j++) {
    d[j] = V[n-1][j];
    V[n-1][j] = 0.0;
  }
  V[n-1][n-1] = 1.0;
  e[0] = 0.0;
} 

// Symmetric tridiagonal QL algorithm.

static void tql2(double V[n][n], double d[n], double e[n]) {

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int i = 1; i < n; i++) {
    e[i-1] = e[i];
  }
  e[n-1] = 0.0;

  double f = 0.0;
  double tst1 = 0.0;
  double eps = pow(2.0,-52.0);
  for (int l = 0; l < n; l++) {

    // Find small subdiagonal element

    tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
    int m = l;
    while (m < n) {
      if (fabs(e[m]) <= eps*tst1) {
        break;
      }
      m++;
    }

    // If m == l, d[l] is an eigenvalue,
    // otherwise, iterate.

    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)

        // Compute implicit shift

        double g = d[l];
        double p = (d[l+1] - g) / (2.0 * e[l]);
        double r = hypot2(p,1.0);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l+1] = e[l] * (p + r);
        double dl1 = d[l+1];
        double h = g - d[l];
        for (int i = l+2; i < n; i++) {
          d[i] -= h;
        }
        f = f + h;

        // Implicit QL transformation.

        p = d[m];
        double c = 1.0;
        double c2 = c;
        double c3 = c;
        double el1 = e[l+1];
        double s = 0.0;
        double s2 = 0.0;
        for (int i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p,e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);

          // Accumulate transformation.

          for (int k = 0; k < n; k++) {
            h = V[k][i+1];
            V[k][i+1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;

        // Check for convergence.

      } while (fabs(e[l]) > eps*tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0;
  }
  
  // Sort eigenvalues and corresponding vectors.

  for (int i = 0; i < n-1; i++) {
    int k = i;
    double p = d[i];
    for (int j = i+1; j < n; j++) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < n; j++) {
        p = V[j][i];
        V[j][i] = V[j][k];
        V[j][k] = p;
      }
    }
  }
}

void eigen_decomposition(double A[n][n], double V[n][n], double d[n]) {
  double e[n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      V[i][j] = A[i][j];
    }
  }
  tred2(V, d, e);
  tql2(V, d, e);
}


PetscErrorCode ACD_read(IBMNodes *ibm, PetscInt ibi, FSInfo *fsi)
{
  PetscInt	rank;
  PetscInt	n_v , n_elmt ;
  PetscReal	*x_bp , *y_bp , *z_bp ;
  PetscInt	*nv1 , *nv2 , *nv3 ;
  PetscReal	*nf_x, *nf_y, *nf_z;
  PetscInt	i,ii;
  PetscInt	n1e, n2e, n3e;
  PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  PetscReal     dr;
  //Added 4/1/06 iman
  PetscReal     *dA ;//area
  PetscReal	*nt_x, *nt_y, *nt_z;
  PetscReal	*ns_x, *ns_y, *ns_z;

  char   ss[20];
  //double xt;
  char string[128];

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(!rank) { // root processor read in the data
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "READ acddata\n");
    char filen[80];  
    sprintf(filen,"./acddata%3.3d" , 0);
 
    fd = fopen(filen, "r"); if (!fd) SETERRQ(1, "Cannot open actuator disc node file")
    n_v =0;

    if (fd) {
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      
      fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      ibm->n_v = n_v;
      ibm->n_elmt = n_elmt;      
      
      MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      //    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
	    
	    /*
      PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
      */
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      x_bp = ibm->x_bp;	// seokkoo
      y_bp = ibm->y_bp;
      z_bp = ibm->z_bp;
      
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
      
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

      for (i=0; i<n_v; i++) {
	fscanf(fd, "%i %le %le %le", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
	
//	x_bp[i] = x_bp[i]*L_dim + CMx_c;//0.25 ;// 24.;	
//	y_bp[i] = y_bp[i]*L_dim + CMy_c ;//+ ibi*2.;//5.;//2.;//8.;//6.;//2.   ;// 24.;
//	z_bp[i] = z_bp[i]*L_dim + CMz_c ;//+ ibi*1.5;//2.;//8.;//15.;//2.   ;// 24.;

        x_bp[i] += fsi->x_c;
        y_bp[i] += fsi->y_c;
        z_bp[i] += fsi->z_c;

	
	ibm->x_bp[i] = x_bp[i];
	ibm->y_bp[i] = y_bp[i];
	ibm->z_bp[i] = z_bp[i];

	ibm->x_bp0[i] = x_bp[i];
	ibm->y_bp0[i] = y_bp[i];
	ibm->z_bp0[i] = z_bp[i];

	ibm->x_bp_o[i] = x_bp[i];
	ibm->y_bp_o[i] = y_bp[i];
	ibm->z_bp_o[i] = z_bp[i];

	ibm->u[i].x = 0.;
	ibm->u[i].y = 0.;
	ibm->u[i].z = 0.;

	ibm->uold[i].x = 0.;
	ibm->uold[i].y = 0.;
	ibm->uold[i].z = 0.;

	ibm->urm1[i].x = 0.;
	ibm->urm1[i].y = 0.;
	ibm->urm1[i].z = 0.;
      }
      i=0;
      PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

/*       ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */
/*       ibm->x_bp_o = x_bp; ibm->y_bp_o = y_bp; ibm->z_bp_o = z_bp; */

      MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
      MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);


      PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
      PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
      PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);
      
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      // Added 4/1/06 iman
      PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      // Added 6/4/06 iman
      //PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));
      // end added

      // added 12-7-2010 xyang
      if (rotor_model) {
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));


        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->i_min));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->j_min));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->k_min));

        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->i_max));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->j_max));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->k_max));

      }

                        if (rotor_model) {      //xyang

                                PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhx));
                                PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhy));
                                PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhz));

                        }


      // end add
      
      
	//seokkoo begin
	{	//only for rank 0
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv1);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv2);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv3);
		
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_x_bp);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_y_bp);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_z_bp);
	}
	
	//seokkoo end

      for (i=0; i<n_elmt; i++) {

	fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
	nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
	      
		// seokkoo
	      ibm->_nv1[i] = nv1[i];
	      ibm->_nv2[i] = nv2[i];
	      ibm->_nv3[i] = nv3[i];
	      // seokkoo
	      ibm->_x_bp[i] = ibm->x_bp[i];
	      ibm->_y_bp[i] = ibm->y_bp[i];
	      ibm->_z_bp[i] = ibm->z_bp[i];

      }
      ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

      i=0;
      PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);

      fclose(fd);
    }
      
    for (i=0; i<n_elmt; i++) {
      
      n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
      dx12 = x_bp[n2e] - x_bp[n1e];
      dy12 = y_bp[n2e] - y_bp[n1e];
      dz12 = z_bp[n2e] - z_bp[n1e];
      
      dx13 = x_bp[n3e] - x_bp[n1e];
      dy13 = y_bp[n3e] - y_bp[n1e];
      dz13 = z_bp[n3e] - z_bp[n1e];
      
      nf_x[i] = dy12 * dz13 - dz12 * dy13;
      nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
      dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
      nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
      // Temp sol. 2D
/*       if (fabs(nf_x[i])<.5) */
/* 	nf_x[i]=0.; */

      // Addedd 4/2/06 iman
      if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
	  (((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
	ns_x[i] = 1.;     
	ns_y[i] = 0.;     
	ns_z[i] = 0. ;
	
	nt_x[i] = 0.;
	nt_y[i] = 1.;
	nt_z[i] = 0.;
      } else {
	ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);      
	ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);     
	ns_z[i] = 0. ;
	
	nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
      }
      
      //Added 4/1/06 iman
      dA[i] = dr/2.; 
      
      // Added 6/4/06 iman
      // Calc the center of the element
      ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
      ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
      ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;	
    }
    
    
    ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
    //Added 4/1/06 iman
    ibm->dA = dA;
    ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
    // Added 4/1/06 iman
    MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    // Added 4/2/06 iman
    MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    // Added 6/4/06 iman
    MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

 /*    PetscFree(dA); */
/*     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); */
/*     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); */
/*     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); */
/*     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3); */
/*     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp); */
    PetscInt ti=0;
    FILE *f;
    //char filen[80];
    sprintf(filen, "./ACDsurface%3.3d_%2.2d_nf.dat",ti,ibi);
    f = fopen(filen, "w");
    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z\n");
    PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
    for (i=0; i<n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
    }
    for (i=0; i<n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
    }
    for (i=0; i<n_v; i++) {	
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
    }
    
/*     PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n"); */
/*     PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, n_elmt); */
/*     for (i=0; i<n_v; i++) { */
      
/*       PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]); */
/*     } */
/*     for (i=0; i<n_elmt; i++) { */
/*       PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1); */
/*     } */
    fclose(f);

  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;
    /*
    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
    */
	
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
	x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
	y_bp = ibm->y_bp;
	z_bp = ibm->z_bp;
	  
    
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
    
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

    for (i=0; i<n_v; i++) {
      ibm->u[i].x = 0.;
      ibm->u[i].y = 0.;
      ibm->u[i].z = 0.;

      ibm->uold[i].x = 0.;
      ibm->uold[i].y = 0.;
      ibm->uold[i].z = 0.;
      
      ibm->urm1[i].x = 0.;
      ibm->urm1[i].y = 0.;
      ibm->urm1[i].z = 0.;      
    }
        
    MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
    MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    

    MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_elmt = n_elmt;

    PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    //Added 4/1/06 iman
    PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

    //Added 4/2/06 iman
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

    ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
    // Added 4/2/06 iman
    ibm->dA = dA;
    ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

    // Added 6/4/06
    //PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

    // added 12-7-2010 xyang
    if (rotor_model) {
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->i_min));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->j_min));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->k_min));

        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->i_max));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->j_max));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->k_max));

    }

                        if (rotor_model) {      //xyang

                                PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhx));
                                PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhy));
                                PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhz));

                        }

    // end add

    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    //Added 4/2/06 iman
    MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  }
	PetscPrintf(PETSC_COMM_WORLD, "Read ACD file !\n");
  return(0);
}



PetscErrorCode Pre_process(UserCtx *user, IBMNodes *ibm)
{
  DA              da = user->da, fda = user->fda;
  DALocalInfo     info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k, l, ibi;
  PetscInt	lxs, lxe, lys, lye, lzs, lze;
  PetscReal	xb_min, xb_max, yb_min, yb_max, zb_min, zb_max;

  Cmpnts	***coor;

  Vec		Coor;

  double hx, hy, hz;

  double xlag1, xlag2;
  double ylag1, ylag2;
  double zlag1, zlag2;

  double xlag, ylag, zlag;

  PetscReal dhx, dhy, dhz;

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


  DAGetGhostedCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor, &coor);
  PetscInt n1e, n2e, n3e;

  for (ibi=0; ibi<NumberOfTurbines; ibi++) {

        for (l=0; l<ibm[ibi].n_v; l++) {

                ibm[ibi].dhx[l] = 0.0;
                ibm[ibi].dhy[l] = 0.0;
                ibm[ibi].dhz[l] = 0.0;

	}


        for (l=0; l<ibm[ibi].n_v; l++) {

                for (i=lxs; i<lxe; i++)
                for (j=lys; j<lye; j++) 
                for (k=lzs; k<lze; k++){ 

                        if ( (ibm[ibi].x_bp[l] >= coor[k][j][i-1].x && ibm[ibi].x_bp[l] <= coor[k][j][i].x) && 
			     (ibm[ibi].y_bp[l] >= coor[k][j-1][i].y && ibm[ibi].y_bp[l] <= coor[k][j][i].y) &&
			     (ibm[ibi].z_bp[l] >= coor[k-1][j][i].z && ibm[ibi].z_bp[l] <= coor[k][j][i].z) ) {
				dhx = coor[k][j][i].x - coor[k][j][i-1].x;
				dhy = coor[k][j][i].y - coor[k][j-1][i].y;
				dhz = coor[k][j][i].z - coor[k-1][j][i].z;

				ibm[ibi].dhx[l] = fabs(dhx);
				ibm[ibi].dhy[l] = fabs(dhy);
				ibm[ibi].dhz[l] = fabs(dhz);

                        }

                }

        }



  	PetscBarrier(PETSC_NULL);

	PetscReal     u;
  	PetscReal     u_sum;
    	for (l=0; l<ibm[ibi].n_v; l++) {

                u = ibm[ibi].dhx[l];
                PetscGlobalSum(&u, &u_sum, PETSC_COMM_WORLD);
                ibm[ibi].dhx[l] = u_sum;

                u = ibm[ibi].dhy[l];
                PetscGlobalSum(&u, &u_sum, PETSC_COMM_WORLD);
                ibm[ibi].dhy[l] = u_sum;

                u = ibm[ibi].dhz[l];
                PetscGlobalSum(&u, &u_sum, PETSC_COMM_WORLD);
                ibm[ibi].dhz[l] = u_sum;

    	}


  }


  for (ibi=0; ibi<NumberOfTurbines; ibi++) {


    	xb_min = 1.0e6; xb_max= -1.0e6;        
    	yb_min = 1.0e6; yb_max= -1.0e6;
    	zb_min = 1.0e6; zb_max= -1.0e6;

    	for (l=0; l<ibm[ibi].n_elmt; l++) {
      		xb_min = PetscMin(xb_min, ibm[ibi].cent_x[l]);
      		xb_max = PetscMax(xb_max, ibm[ibi].cent_x[l]);

      		yb_min = PetscMin(yb_min, ibm[ibi].cent_y[l]);
      		yb_max = PetscMax(yb_max, ibm[ibi].cent_y[l]);

      		zb_min = PetscMin(zb_min, ibm[ibi].cent_z[l]);
      		zb_max = PetscMax(zb_max, ibm[ibi].cent_z[l]);
    	}

//    PetscPrintf(PETSC_COMM_SELF, "**** the xb_min, xb_max: %le %le \n", xb_min, xb_max);
//    PetscPrintf(PETSC_COMM_SELF, "**** the yb_min, yb_max: %le %le \n", yb_min, yb_max);
//    PetscPrintf(PETSC_COMM_SELF, "**** the zb_min, zb_max: %le %le \n", zb_min, zb_max);
        int iii1 = 0;
	int iii2 = 0;
        int iii = 0;

        for (l=0; l<ibm[ibi].n_elmt; l++) {

      		n1e = ibm[ibi].nv1[l]; n2e = ibm[ibi].nv2[l]; n3e = ibm[ibi].nv3[l];

		dhx = max(max(ibm[ibi].dhx[n1e], ibm[ibi].dhx[n2e]), ibm[ibi].dhx[n3e]);		
		dhy = max(max(ibm[ibi].dhy[n1e], ibm[ibi].dhy[n2e]), ibm[ibi].dhy[n3e]);		
		dhz = max(max(ibm[ibi].dhz[n1e], ibm[ibi].dhz[n2e]), ibm[ibi].dhz[n3e]);		

                xlag1 = ibm[ibi].cent_x[l] - 4.0*dhx; xlag2 = ibm[ibi].cent_x[l] + 4.0*dhx;

                ylag1 = ibm[ibi].cent_y[l] - 4.0*dhy; ylag2 = ibm[ibi].cent_y[l] + 4.0*dhy;

                zlag1 = ibm[ibi].cent_z[l] - 4.0*dhz; zlag2 = ibm[ibi].cent_z[l] + 4.0*dhz;


		if (rotor_model) {
	                xlag1 = xb_min - 4.0*dhx; xlag2 = xb_max + 4.0*dhx;

                	ylag1 = yb_min - 4.0*dhy; ylag2 = yb_max + 4.0*dhy;

                	zlag1 = zb_min - 4.0*dhz; zlag2 = zb_max + 4.0*dhz;
		}


		iii1 = 0;
		iii2 = 0;
		iii  = 0;

		j = lys; k =lzs;
                for (i=lxs; i<lxe; i++){

 
                        if (xlag1 >= coor[k][j][i-1].x && xlag1 <= coor[k][j][i].x) { 
                        	ibm[ibi].i_min[l] = i;
                        	iii1 = 1;
                        }
                        if (xlag2 >= coor[k][j][i-1].x && xlag2 <= coor[k][j][i].x) {
                                ibm[ibi].i_max[l] = i;
                                iii2 = 1;
                        }
//                        if (ibm[ibi].cent_x[l] >= coor[k][j][i-1].x && ibm[ibi].cent_x[l] <= coor[k][j][i].x) {
//                                iii = 1;
//                        }

                }

                if (iii1 && !iii2) {
                        ibm[ibi].i_max[l] = lxe;
                }
                if (!iii1 && iii2) {
                        ibm[ibi].i_min[l] = lxs;
                }
                if (!iii1 && !iii2) {
                        ibm[ibi].i_min[l] = lxs;
                        ibm[ibi].i_max[l] = lxs;
                }


		if (xlag1 <= coor[k][j][lxs].x &&  xlag2 >= coor[k][j][lxe-1].x) {
			ibm[ibi].i_min[l] = lxs;
			ibm[ibi].i_max[l] = lxe;
		}


//		if (i_periodicIB && ii_periodic && ibm[ibi].i_min[l] == xs+1) ibm[ibi].i_min[l] = xs - 2;
//		if (i_periodicIB && ii_periodic && ibm[ibi].i_max[l] == xe-1) ibm[ibi].i_max[l] = xe + 2;


		iii1 = 0;
		iii2 = 0;
		iii  = 0;

		i = lxs; k = lzs;
                for (j=lys; j<lye; j++){ 

			if (ylag1 >= coor[k][j-1][i].y && ylag1 <= coor[k][j][i].y) { 
                        	ibm[ibi].j_min[l] = j;
                        	iii1 = 1;
                        }
                        if (ylag2 >= coor[k][j-1][i].y && ylag2 <= coor[k][j][i].y) {
                                ibm[ibi].j_max[l] = j;
                                iii2 = 1;
                        }
                }


                if (iii1 && !iii2) {
                        ibm[ibi].j_max[l] = lye;
                }
                if (!iii1 && iii2) {
                        ibm[ibi].j_min[l] = lys;
                }
                if (!iii1 && !iii2) {
                        ibm[ibi].j_min[l] = lys;
                        ibm[ibi].j_max[l] = lys;
                }

                if (ylag1 <= coor[k][lys][i].y &&  ylag2 >= coor[k][lye-1][i].y) {
                        ibm[ibi].j_min[l] = lys;
                        ibm[ibi].j_max[l] = lye;
                }

//                if (j_periodicIB && jj_periodic && ibm[ibi].j_min[l] == ys+1) ibm[ibi].j_min[l] = ys - 2;
//                if (j_periodicIB && jj_periodic && ibm[ibi].j_max[l] == ye-1) ibm[ibi].j_max[l] = ye + 2;


		
		iii1 = 0;
		iii2 = 0;
		iii  = 0;


		i = lxs; j = lys;
                for (k=lzs; k<lze; k++){ 

                        if (zlag1 >= coor[k-1][j][i].z && zlag1 <= coor[k][j][i].z) { 
                        	ibm[ibi].k_min[l] = k;
                        	iii1 = 1;
                        }
                        if (zlag2 >= coor[k-1][j][i].z && zlag2 <= coor[k][j][i].z) {
                                ibm[ibi].k_max[l] = k;
                                iii2 = 1;
                        }
                }


                if (iii1 && !iii2) {
                        ibm[ibi].k_max[l] = lze;
                }
                if (!iii1 && iii2) {
                        ibm[ibi].k_min[l] = lzs;
                }
                if (!iii1 && !iii2) {
                        ibm[ibi].k_min[l] = lzs;
                        ibm[ibi].k_max[l] = lzs;
                }


                if (zlag1 <= coor[lzs][j][i].z &&  zlag2 >= coor[lze-1][j][i].z) {
                        ibm[ibi].k_min[l] = lzs;
                        ibm[ibi].k_max[l] = lze;
                }


//                if (k_periodicIB && kk_periodic && ibm[ibi].k_min[l] == zs+1) ibm[ibi].k_min[l] = zs - 2;
//                if (k_periodicIB && kk_periodic && ibm[ibi].k_max[l] == ze-1) ibm[ibi].k_max[l] = ze + 2;

	
//		        ibm[ibi].i_min[l] = lxs;
//                        ibm[ibi].i_max[l] = lxe;

//                        ibm[ibi].j_min[l] = lys;
//                        ibm[ibi].j_max[l] = lye;

//                        ibm[ibi].k_min[l] = lzs;
//                        ibm[ibi].k_max[l] = lze;


        }

//	l = 0;
//        printf("i_min, i_max %d %d \n", ibm[ibi].i_min[l], ibm[ibi].i_max[l]);
//        printf("j_min, j_max %d %d \n", ibm[ibi].j_min[l], ibm[ibi].j_max[l]);
//        printf("k_min, k_max %d %d \n", ibm[ibi].k_min[l], ibm[ibi].k_max[l]);
//	}

  }


  DAVecRestoreArray(fda, Coor, &coor);

  return(0);

}


/* Interpolate the velocity at the Lagrangian points */
PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, Vec v_eul, char fname[80])
{

  DA              da = user->da, fda = user->fda;
  DALocalInfo     info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k, l, ibi;
  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  Cmpnts	***ucat, ***coor, ***csi, ***eta, ***zet;

  Vec		Coor;

  PetscReal	dfunc;

  PetscReal	xc, yc, zc, hx, hy, hz, vol_eul, vol_lagr;

  double r1, r2, r3;

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


  DAGetGhostedCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor, &coor);
//  DAVecGetArray(fda, user->lUcat, &ucat);
  DAVecGetArray(fda, v_eul, &ucat);
//  DAVecGetArray(fda, user->lCsi,  &csi);
//  DAVecGetArray(fda, user->lEta,  &eta);
//  DAVecGetArray(fda, user->lZet,  &zet);

  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    for (l=0; l<ibm[ibi].n_elmt; l++) {
      ibm[ibi].U_lagr_x[l] = 0.0;
      ibm[ibi].U_lagr_y[l] = 0.0;
      ibm[ibi].U_lagr_z[l] = 0.0;
    }
  }


//        clock_t start, end;
//        double elapsed;

//        start = clock();


  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
//        for (l=0; l<ibm[ibi].n_elmt; l++) {
//        printf("i_min, i_max %d %d \n", ibm[ibi].i_min[l], ibm[ibi].i_max[l]);
//        printf("j_min, j_max %d %d \n", ibm[ibi].j_min[l], ibm[ibi].j_max[l]);
//        printf("k_min, k_max %d %d \n", ibm[ibi].k_min[l], ibm[ibi].k_max[l]);
//        }

  for (l=0; l<ibm[ibi].n_elmt; l++) {
    	for (k = ibm[ibi].k_min[l]; k<ibm[ibi].k_max[l]; k++) 
      	for (j = ibm[ibi].j_min[l]; j<ibm[ibi].j_max[l]; j++) 
        for (i = ibm[ibi].i_min[l]; i<ibm[ibi].i_max[l]; i++) {


//        for (k = lzs; k<lze; k++)
//        for (j = lys; j<lye; j++)
//        for (i = lxs; i<lxe; i++) {

         	xc = (coor[k][j][i  ].x + coor[k][j-1][i  ].x + coor[k-1][j][i  ].x + coor[k-1][j-1][i  ].x +
                	coor[k][j][i-1].x + coor[k][j-1][i-1].x + coor[k-1][j][i-1].x + coor[k-1][j-1][i-1].x ) * 0.125;
          	yc = (coor[k][j  ][i].y + coor[k][j  ][i-1].y + coor[k-1][j  ][i].y + coor[k-1][j  ][i-1].y + 
                	coor[k][j-1][i].y + coor[k][j-1][i-1].y + coor[k-1][j-1][i].y + coor[k-1][j-1][i-1].y ) * 0.125;
          	zc = (coor[k  ][j][i].z + coor[k  ][j-1][i].z + coor[k  ][j][i-1].z + coor[k  ][j-1][i-1].z + 
                	coor[k-1][j][i].z + coor[k-1][j-1][i].z + coor[k-1][j][i-1].z + coor[k-1][j-1][i-1].z ) * 0.125;

          	hx = (coor[k][j][i  ].x + coor[k][j-1][i  ].x + coor[k-1][j][i  ].x + coor[k-1][j-1][i  ].x -
                	coor[k][j][i-1].x - coor[k][j-1][i-1].x - coor[k-1][j][i-1].x - coor[k-1][j-1][i-1].x ) * 0.25;

          	hy = (coor[k][j  ][i].y + coor[k][j  ][i-1].y + coor[k-1][j  ][i].y + coor[k-1][j  ][i-1].y -
                	coor[k][j-1][i].y - coor[k][j-1][i-1].y - coor[k-1][j-1][i].y - coor[k-1][j-1][i-1].y ) * 0.25;

          	hz = (coor[k  ][j][i].z + coor[k  ][j-1][i].z + coor[k  ][j][i-1].z + coor[k  ][j-1][i-1].z - 
                	coor[k-1][j][i].z - coor[k-1][j-1][i].z - coor[k-1][j][i-1].z - coor[k-1][j-1][i-1].z ) * 0.25;

            	r1 = (xc - ibm[ibi].cent_x[l])/hx; r2 = (yc - ibm[ibi].cent_y[l])/hy; r3 = (zc - ibm[ibi].cent_z[l])/hz;
            	dfunc =  dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);

//		if (deltafunc == 1) dfunc =  dfunc_2h(r1) * dfunc_2h(r2) * dfunc_2h(r3);
//		if (deltafunc == 2) dfunc =  dfunc_4h(r1) * dfunc_4h(r2) * dfunc_4h(r3);

//		if (IB_TwoD) {

//	                dfunc =  dfunc_s3h(r2) * dfunc_s3h(r3) / ((double) (xe-2));
//        	        if (deltafunc == 1) dfunc =  dfunc_2h(r2) * dfunc_2h(r3) / ((double) (xe-2));
//                	if (deltafunc == 2) dfunc =  dfunc_4h(r2) * dfunc_4h(r3) / ((double) (xe-2));

//		}

            	ibm[ibi].U_lagr_x[l] += ucat[k][j][i].x * dfunc;
            	ibm[ibi].U_lagr_y[l] += ucat[k][j][i].y * dfunc;
            	ibm[ibi].U_lagr_z[l] += ucat[k][j][i].z * dfunc;          
   
      	}
  }
  }

//        end = clock();
//        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

        PetscPrintf(PETSC_COMM_WORLD, "Average over disk \n");


//  PetscPrintf(PETSC_COMM_SELF, "the U_lagr_z: %le \n", ibm[0].U_lagr_z[0] );

//        start = clock();

  PetscBarrier(PETSC_NULL);

// MPI_Allreduce( &LM_tmp[0], &J_LM[0], my, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

 // the n_elmt for each turbine is assumed to be the same

  double    u_local[NumberOfTurbines][ibm[0].n_elmt] ,u_sum[NumberOfTurbines][ibm[0].n_elmt];
  int totNum = NumberOfTurbines*ibm[0].n_elmt;

// x
  for (ibi=0; ibi<NumberOfTurbines; ibi++) 
  for (i=0; i<ibm[ibi].n_elmt; i++ ) {
	u_local[ibi][i] = ibm[ibi].U_lagr_x[i];
  }

  MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  for (ibi=0; ibi<NumberOfTurbines; ibi++) 
  for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        ibm[ibi].U_lagr_x[i] = u_sum[ibi][i];
  }

  PetscBarrier(PETSC_NULL);
// y
  for (ibi=0; ibi<NumberOfTurbines; ibi++) 
  for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        u_local[ibi][i] = ibm[ibi].U_lagr_y[i];
  }

  MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  for (ibi=0; ibi<NumberOfTurbines; ibi++)
  for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        ibm[ibi].U_lagr_y[i] = u_sum[ibi][i];
  }

  PetscBarrier(PETSC_NULL);
// z
  for (ibi=0; ibi<NumberOfTurbines; ibi++) 
  for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        u_local[ibi][i] = ibm[ibi].U_lagr_z[i];
  }

  MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  for (ibi=0; ibi<NumberOfTurbines; ibi++)
  for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        ibm[ibi].U_lagr_z[i] = u_sum[ibi][i];
  }


/*

  PetscReal	u_sum;
  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    for (l=0; l<ibm[ibi].n_elmt; l++) {
      PetscGlobalSum(&(ibm[ibi].U_lagr_x[l]), &u_sum, PETSC_COMM_WORLD);
      ibm[ibi].U_lagr_x[l] = u_sum;      

      PetscGlobalSum(&(ibm[ibi].U_lagr_y[l]), &u_sum, PETSC_COMM_WORLD);
      ibm[ibi].U_lagr_y[l] = u_sum;

      PetscGlobalSum(&(ibm[ibi].U_lagr_z[l]), &u_sum, PETSC_COMM_WORLD);
      ibm[ibi].U_lagr_z[l] = u_sum;

//      printf(" the ulagr %le \n", ibm[ibi].U_lagr_z[l]);
    }
  }
*/
//        end = clock();
//        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

//        PetscPrintf(PETSC_COMM_WORLD, "Time for U_larg global sum %le \n", elapsed);


  DAVecRestoreArray(fda, Coor, &coor);
  DAVecRestoreArray(fda, v_eul, &ucat);
//  DAVecRestoreArray(fda, user->lUcat, &ucat);
//  DAVecRestoreArray(fda, user->lCsi,  &csi);
//  DAVecRestoreArray(fda, user->lEta,  &eta);
//  DAVecRestoreArray(fda, user->lZet,  &zet);

  PetscReal U[ibm[0].n_elmt], V[ibm[0].n_elmt], W[ibm[0].n_elmt];

    for (l=0; l<ibm[0].n_elmt; l++) {
	U[l] = 0.0;
	V[l] = 0.0;
	W[l] = 0.0;
    }

  	for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    	for (l=0; l<ibm[0].n_elmt; l++) {
		U[l] += ibm[ibi].U_lagr_x[l];
		V[l] += ibm[ibi].U_lagr_y[l];
		W[l] += ibm[ibi].U_lagr_z[l];
    	}
  	}


    for (l=0; l<ibm[0].n_elmt; l++) {
        U[l] = U[l] / double(NumberOfTurbines);
        V[l] = V[l] / double(NumberOfTurbines);
        W[l] = W[l] / double(NumberOfTurbines);
    }

  // output the value on the disc

    FILE *f;
    //char filen[80];
    f = fopen(fname, "w");
    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,U,V,W\n");
    PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", ibm[0].n_v, ibm[0].n_elmt);
    for (i=0; i<ibm[0].n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm[0].x_bp[i]);
    }
    for (i=0; i<ibm[0].n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm[0].y_bp[i]);
    }
    for (i=0; i<ibm[0].n_v; i++) {	
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm[0].z_bp[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", U[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", V[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", W[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm[0].nv1[i]+1, ibm[0].nv2[i]+1, ibm[0].nv3[i]+1);
    }
    
    fclose(f);




  double sum_x[NumberOfTurbines], sum_y[NumberOfTurbines], sum_z[NumberOfTurbines];

  double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

  double fac = 1.0 / double(NumberOfTurbines);  

  double A_sum;

  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    sum_x[ibi] = 0.0;
    sum_y[ibi] = 0.0;
    sum_z[ibi] = 0.0;
    A_sum = 0.0;
    for (l=0; l<ibm[ibi].n_elmt; l++) {
      sum_x[ibi] += ibm[ibi].U_lagr_x[l] * ibm[ibi].dA[l];
      sum_y[ibi] += ibm[ibi].U_lagr_y[l] * ibm[ibi].dA[l];
      sum_z[ibi] += ibm[ibi].U_lagr_z[l] * ibm[ibi].dA[l];
      A_sum += ibm[ibi].dA[l];
    }
    sum_x[ibi] /= A_sum;
    sum_y[ibi] /= A_sum;
    sum_z[ibi] /= A_sum;

    sum1 +=  sum_x[ibi] * fac;
    sum2 +=  sum_y[ibi] * fac;
    sum3 +=  sum_z[ibi] * fac;
  }

//	FILE *f;
      char filen[80];
//      filen = fname+".dat";
      sprintf(filen, fname);
      f = fopen(filen, "a");
      PetscFPrintf(PETSC_COMM_WORLD, f, "The mean uu on the disc: %le \n", sum1);
      PetscFPrintf(PETSC_COMM_WORLD, f, "The mean vv on the disc: %le \n", sum2);
      PetscFPrintf(PETSC_COMM_WORLD, f, "The mean ww on the disc: %le \n", sum3);
      fclose(f);

  printf("The mean uu on the disc: %le \n", sum1);
  printf("The mean vv on the disc: %le \n", sum2);
  printf("The mean ww on the disc: %le \n", sum3);


  return(0);

}



/* Interpolate the velocity at the Lagrangian points */
/*
PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, Vec v_eul, char fname[80])
{

  DA              da = user->da, fda = user->fda;
  DALocalInfo     info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k, l, ibi;
  PetscInt	lxs, lxe, lys, lye, lzs, lze;

  Cmpnts	***ucat, ***coor, ***csi, ***eta, ***zet;

  Vec		Coor;

  PetscReal	dfunc;

  PetscReal	xc, yc, zc, hx, hy, hz, vol_eul, vol_lagr;

  double r1, r2, r3;

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


  DAGetGhostedCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor, &coor);
  DAVecGetArray(fda, v_eul, &ucat);

  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    for (l=0; l<ibm[ibi].n_elmt; l++) {
      ibm[ibi].U_lagr_x[l] = 0.0;
      ibm[ibi].U_lagr_y[l] = 0.0;
      ibm[ibi].U_lagr_z[l] = 0.0;
    }
  }

  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
//        for (l=0; l<ibm[ibi].n_elmt; l++) {
//        printf("i_min, i_max %d %d \n", ibm[ibi].i_min[l], ibm[ibi].i_max[l]);
//        printf("j_min, j_max %d %d \n", ibm[ibi].j_min[l], ibm[ibi].j_max[l]);
//        printf("k_min, k_max %d %d \n", ibm[ibi].k_min[l], ibm[ibi].k_max[l]);
//        }


  for (l=0; l<ibm[ibi].n_elmt; l++) {
        for (k = ibm[ibi].k_min[l]; k<ibm[ibi].k_max[l]; k++)
        for (j = ibm[ibi].j_min[l]; j<ibm[ibi].j_max[l]; j++)
        for (i = ibm[ibi].i_min[l]; i<ibm[ibi].i_max[l]; i++) {


          xc = (coor[k][j][i  ].x + coor[k][j-1][i  ].x + coor[k-1][j][i  ].x + coor[k-1][j-1][i  ].x +
                coor[k][j][i-1].x + coor[k][j-1][i-1].x + coor[k-1][j][i-1].x + coor[k-1][j-1][i-1].x ) * 0.125;
          yc = (coor[k][j  ][i].y + coor[k][j  ][i-1].y + coor[k-1][j  ][i].y + coor[k-1][j  ][i-1].y + 
                coor[k][j-1][i].y + coor[k][j-1][i-1].y + coor[k-1][j-1][i].y + coor[k-1][j-1][i-1].y ) * 0.125;
          zc = (coor[k  ][j][i].z + coor[k  ][j-1][i].z + coor[k  ][j][i-1].z + coor[k  ][j-1][i-1].z + 
                coor[k-1][j][i].z + coor[k-1][j-1][i].z + coor[k-1][j][i-1].z + coor[k-1][j-1][i-1].z ) * 0.125;

	  xc = coor[k][j][i  ].x;
	  yc = coor[k][j  ][i].y;
	  zc = coor[k][j  ][i].z;

          hx = (coor[k][j][i  ].x + coor[k][j-1][i  ].x + coor[k-1][j][i  ].x + coor[k-1][j-1][i  ].x -
                coor[k][j][i-1].x - coor[k][j-1][i-1].x - coor[k-1][j][i-1].x - coor[k-1][j-1][i-1].x ) * 0.25;

          hy = (coor[k][j  ][i].y + coor[k][j  ][i-1].y + coor[k-1][j  ][i].y + coor[k-1][j  ][i-1].y -
                coor[k][j-1][i].y - coor[k][j-1][i-1].y - coor[k-1][j-1][i].y - coor[k-1][j-1][i-1].y ) * 0.25;

          hz = (coor[k  ][j][i].z + coor[k  ][j-1][i].z + coor[k  ][j][i-1].z + coor[k  ][j-1][i-1].z - 
                coor[k-1][j][i].z - coor[k-1][j-1][i].z - coor[k-1][j][i-1].z - coor[k-1][j-1][i-1].z ) * 0.25;

          for (l=0; l<ibm[ibi].n_elmt; l++) {
            r1 = (xc - ibm[ibi].cent_x[l])/hx; r2 = (yc - ibm[ibi].cent_y[l])/hy; r3 = (zc - ibm[ibi].cent_z[l])/hz;
            dfunc =  dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
            ibm[ibi].U_lagr_x[l] += ucat[k][j][i].x * dfunc;
            ibm[ibi].U_lagr_y[l] += ucat[k][j][i].y * dfunc;
            ibm[ibi].U_lagr_z[l] += ucat[k][j][i].z * dfunc;

          }
        }
    }
  }

//  PetscPrintf(PETSC_COMM_SELF, "the U_lagr_z: %le \n", ibm[0].U_lagr_z[0] );
  PetscBarrier(PETSC_NULL);

  PetscReal	u_sum;
  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    for (l=0; l<ibm[ibi].n_elmt; l++) {
      PetscGlobalSum(&(ibm[ibi].U_lagr_x[l]), &u_sum, PETSC_COMM_WORLD);
      ibm[ibi].U_lagr_x[l] = u_sum;      

      PetscGlobalSum(&(ibm[ibi].U_lagr_y[l]), &u_sum, PETSC_COMM_WORLD);
      ibm[ibi].U_lagr_y[l] = u_sum;

      PetscGlobalSum(&(ibm[ibi].U_lagr_z[l]), &u_sum, PETSC_COMM_WORLD);
      ibm[ibi].U_lagr_z[l] = u_sum;

//      printf(" the ulagr %le \n", ibm[ibi].U_lagr_z[l]);
    }
  }

  DAVecRestoreArray(fda, Coor, &coor);
  DAVecRestoreArray(fda, v_eul, &ucat);


  PetscReal U[ibm[0].n_elmt], V[ibm[0].n_elmt], W[ibm[0].n_elmt];

    for (l=0; l<ibm[0].n_elmt; l++) {
	U[l] = 0.0;
	V[l] = 0.0;
	W[l] = 0.0;
    }

  	for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    	for (l=0; l<ibm[0].n_elmt; l++) {
		U[l] += ibm[ibi].U_lagr_x[l];
		V[l] += ibm[ibi].U_lagr_y[l];
		W[l] += ibm[ibi].U_lagr_z[l];
    	}
  	}


    for (l=0; l<ibm[0].n_elmt; l++) {
        U[l] = U[l] / double(NumberOfTurbines);
        V[l] = V[l] / double(NumberOfTurbines);
        W[l] = W[l] / double(NumberOfTurbines);
    }

  // output the value on the disc

    FILE *f;
    //char filen[80];
    f = fopen(fname, "w");
    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,U,V,W\n");
    PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", ibm[0].n_v, ibm[0].n_elmt);
    for (i=0; i<ibm[0].n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm[0].x_bp[i]);
    }
    for (i=0; i<ibm[0].n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm[0].y_bp[i]);
    }
    for (i=0; i<ibm[0].n_v; i++) {	
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm[0].z_bp[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", U[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", V[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", W[i]);
    }
    for (i=0; i<ibm[0].n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm[0].nv1[i]+1, ibm[0].nv2[i]+1, ibm[0].nv3[i]+1);
    }
    
    fclose(f);




  double sum_x[NumberOfTurbines], sum_y[NumberOfTurbines], sum_z[NumberOfTurbines];

  double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

  double fac = 1.0 / double(NumberOfTurbines);  

  double A_sum;

  for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    sum_x[ibi] = 0.0;
    sum_y[ibi] = 0.0;
    sum_z[ibi] = 0.0;
    A_sum = 0.0;
    for (l=0; l<ibm[ibi].n_elmt; l++) {
      sum_x[ibi] += ibm[ibi].U_lagr_x[l] * ibm[ibi].dA[l];
      sum_y[ibi] += ibm[ibi].U_lagr_y[l] * ibm[ibi].dA[l];
      sum_z[ibi] += ibm[ibi].U_lagr_z[l] * ibm[ibi].dA[l];
      A_sum += ibm[ibi].dA[l];
    }
    sum_x[ibi] /= A_sum;
    sum_y[ibi] /= A_sum;
    sum_z[ibi] /= A_sum;

    sum1 +=  sum_x[ibi] * fac;
    sum2 +=  sum_y[ibi] * fac;
    sum3 +=  sum_z[ibi] * fac;
  }

//	FILE *f;
      char filen[80];
//      filen = fname+".dat";
      sprintf(filen, fname);
      f = fopen(filen, "a");
      PetscFPrintf(PETSC_COMM_WORLD, f, "The mean uu on the disc: %le \n", sum1);
      PetscFPrintf(PETSC_COMM_WORLD, f, "The mean vv on the disc: %le \n", sum2);
      PetscFPrintf(PETSC_COMM_WORLD, f, "The mean ww on the disc: %le \n", sum3);
      fclose(f);

  printf("The mean uu on the disc: %le \n", sum1);
  printf("The mean vv on the disc: %le \n", sum2);
  printf("The mean ww on the disc: %le \n", sum3);


  return(0);

}

*/

double dfunc_4h(double r)
{
  if (fabs(r) <= 1.0) {
    return (3.0-2.0*fabs(r)+sqrt(1.0+4.0*fabs(r)-4.0*r*r))/8.0;
  } else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    return (5.0-2.0*fabs(r)-sqrt(-7.0+12.0*fabs(r)-4.0*r*r))/8.0;
  } else {
    return 0.0;
  }
}

double dfunc_s3h(double r)
{
  if (fabs(r) <= 1.0) {
    return 17.0/48.0 + sqrt(3.0)*3.14159265/108.0 + fabs(r)/4.0 - r*r/4.0 + (1.0-2.0*fabs(r))*sqrt(-12.0*r*r+12.0*fabs(r)+1.0)/16.0 - sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-1.0)/2.0)/12.0;
  } else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    return 55.0/48.0 - sqrt(3.0)*3.14159265/108.0 - 13.0*fabs(r)/12.0 + r*r/4.0 + (2.0*fabs(r)-3.0)*sqrt(-12.0*r*r+36.0*fabs(r)-23.0)/48.0 + sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-3.0)/2.0)/36.0;
  } else { 
    return 0.0;
  }

}


double dfunc_2h(double r)
{
  if (fabs(r) < 1.0) {
    return 1.0-fabs(r);
  } else {
    return 0.0;
  }
}

