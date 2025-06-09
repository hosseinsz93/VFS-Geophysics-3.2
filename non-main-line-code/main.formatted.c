#include <string>

#include "variables.h"
static char help[] = "Testing programming!";
PetscReal COEF_TIME_ACCURACY=1.5;
PetscInt ti,tistart=0;
PetscReal Flux_in = 4.104388e-04, angle = 0;
PetscInt tiout = 10;
int block_number=1;
PetscReal FluxInSum, FluxOutSum;
PetscReal FluxInSum_gas, FluxOutSum_gas;	// seokkoo
PetscInt immersed = 0;
PetscInt inviscid = 0;
PetscInt movefsi = 0, rotatefsi=0;
PetscInt implicit = 0;
PetscInt imp_MAX_IT = 50;
PetscInt radi=10;
PetscInt inletprofile=1;
PetscReal CMx_c=0., CMy_c=0., CMz_c=0.;
PetscInt  mg_MAX_IT=30, mg_idx=1, mg_preItr=1, mg_poItr=1;
PetscReal imp_atol=1e-7, imp_rtol=1e-4, imp_stol=1.e-8;
PetscInt TwoD = 0;
PetscInt STRONG_COUPLING=0;
PetscInt rstart_fsi=0;
PetscInt cop=0, regime=1; // 1 escape regime --- 2 cruise regime
PetscInt fish=0;
PetscReal St_exp=0.5,wavelength=0.95;
PetscInt MHV=0;
PetscReal max_angle = -54.*3.1415926/180.;// for MHV; min=0
PetscInt thin=0;
PetscInt dgf_z=0,dgf_y=1,dgf_x=0;
PetscInt dgf_az=0,dgf_ay=0,dgf_ax=1 ;
PetscInt NumberOfBodies=1, NumberOfRotatingBodies=1;
int block_numbe;
PetscReal L_dim;
PetscInt averaging=0;
PetscInt phase_averaging=0;
PetscInt binary_input=0;
PetscInt xyz_input=0;
PetscInt les=0;
PetscInt inlet_buffer_k=1;
PetscInt wallfunction=0;
PetscInt slipbody=-1000;
PetscInt central=0, second_order=0, weno = 0;
PetscInt initialzero=0;
PetscInt freesurface=0;
PetscInt rans=0, lowRe=0;
PetscInt cross_diffusion=1;
PetscInt surface_tension=0;
PetscReal Background_Conc=0;
PetscReal Inlet_Conc=0;
PetscInt zero_grad=1;
PetscInt subdt_levelset=1;
double dt_inflow;

PetscInt levelset=0;
PetscInt sloshing=0;
PetscInt bubble=0;
double sloshing_a=0.05, sloshing_b=2, sloshing_d=1;
double bubble_d=1., bubble_z=0., bubble_ws=10.;
PetscInt fix_level=0;
PetscInt laplacian=0;
PetscInt qcr=0;
PetscInt poisson=1;
PetscInt amg_agg=1;
double amg_thresh=0.3;
PetscInt amg_coarsentype=8;
PetscInt periodic=0;
PetscInt i_periodic=0;
PetscInt ii_periodic=0;
PetscInt j_periodic=0;
PetscInt jj_periodic=0;
PetscInt k_periodic=0;
PetscInt kk_periodic=0;
PetscInt pseudo_periodic=0;
double inlet_flux=-1;
PetscInt delete_previous_file=0;
PetscInt mixed=0;
PetscInt clark=0;
PetscInt vorticity=0;
PetscInt initial_perturbation=0;
PetscInt skew=0;
PetscInt dynamic_freq=1;
double fluct_rms=0.005;
int my_rank;
int ib_bctype[128];
char path[256], gridfile[256];
PetscInt i_proc=PETSC_DECIDE, j_proc=PETSC_DECIDE, k_proc=PETSC_DECIDE;
double imp_free_tol=1.e-4;
double poisson_tol=5.e-9;	// relative tolerance
PetscReal les_eps=1.e-7;
double water_vol=0, water_vol_o=0;

//double Fr=1.0;

double mean_pressure_gradient=0;	// relative tolerance
PetscReal max_cs=0.5;
PetscBool dpdz_set=PETSC_FALSE;
PetscInt save_inflow=0;
PetscInt save_inflow_period=100;
PetscInt read_inflow_period=100;
PetscInt save_inflow_minus=0;
int save_ksection[1000];
int nsave_ksection=0;
int ucat_plane_allocated=0;

PetscInt ti_lastsave=0;
PetscInt localstep=1;
PetscInt inflow_recycle_perioid=20000;
PetscInt save_memory=1;
PetscInt ibm_search=0;
PetscBool rough_set=PETSC_FALSE;
double roughness_size=0.0;
int save_point[3000][10];
double save_coor[3000][10];  // xiaolei
int nsave_points=0;

int save_IBpoint[3000][10];  // xiaolei
double save_IBcoor[3000][10];  // xiaolei
int nsave_IBpoints=0; 	// xiaolei


PetscInt testfilter_ik=0;
PetscInt testfilter_1d=0;
PetscInt i_homo_filter=0;
PetscInt j_homo_filter=0;
PetscInt k_homo_filter=0;
PetscInt poisson_it=10;
PetscInt tiout_ufield = -1, tiend_ufield = 10000000;
//PetscInt display_implicit_count=0;
double dx_min, di_min, dj_min, dk_min;
double di_max, dj_max, dk_max;

double rho_water=1000., rho_air=1.204;	// bubble
double dthick_const=-1;
//double mu_water=1, mu_air=0.1;
double dthick=1.0, dtau_ratio=0.2, d_parameter=0.25;
PetscBool dthick_set=PETSC_FALSE;
//double rho_water=1., rho_air=0.001;
double mu_water=1.e-3, mu_air=1.78e-5;
double angvel=3.141592;
//double rho_water=10., rho_air=0.01;

double gravity_x=0, gravity_y=0, gravity_z=0;
double inlet_y=0, outlet_y=0;
double inlet_z=0, outlet_z=0;
PetscInt fix_outlet=0, fix_inlet=0;
PetscInt levelset_it=8;
PetscInt rotdir=2; // 0: rotate around the x-axis, 1:y-axis, 2:z-axis
double x_r=0, y_r=0, z_r=0; // center of rotation of rfsi

// add (xiaolei)
PetscInt imin_wm=0, imax_wm=0, jmin_wm=0, jmax_wm=0, kmin_wm=0, kmax_wm=0;
PetscInt imin_wmtmprt=0, imax_wmtmprt=0, jmin_wmtmprt=0, jmax_wmtmprt=0, kmin_wmtmprt=0, kmax_wmtmprt=0;

PetscInt NumberOfTurbines=1; //xyang
PetscInt NumberOfIBDelta=1; //xyang
PetscInt rotatewt=0; //xyang
PetscInt IB_delta = 0; // xyang
PetscInt NumIBPerLoc = 1;

PetscInt IB_wm = 0;
PetscInt IB_wmtmprt = 0;
PetscInt i_periodicIB = 0, j_periodicIB = 0, k_periodicIB = 0;
PetscInt ApproxBC_wm = 0;
PetscInt Shear_wm = 1;
PetscInt Force_wm = 0;
PetscInt infRe = 0;

double xmin,xmax,ymin,ymax,zmin,zmax; // the range of domain used for moving frame with wind turbines

PetscInt rotor_model = 0;  // xyang 12-7-2010 1: actuator disk model 2, 3: actuator line model
PetscReal indf_ax = 0.25; // xyang 12-16-2010
PetscInt surface_p_out = 0; //xyang
PetscInt num_blade = 3; // xyang 1-21-2011
PetscInt num_foiltype = 2; // The number of airfoil types used in the blade xyang 03-18-2011
PetscReal dh1_wm = 0.001; // the first off-wall grid spacing for wall model 4-11-2011
PetscReal dhratio_wm = 1.05; // the ratio of grid spacing in wall model
PetscReal reflength_wt = 1.0;
PetscReal reflength_IBDelta = 1.0;
PetscInt temperature = 0;
PetscInt deltafunc = 3;
PetscInt add_fluctuations = 0;
PetscInt add_fluctuations_tmprt = 0;
PetscReal tipspeedratio = 4.1;
PetscReal r_rotor = 0.05;
PetscReal r_nacelle = 0.05;

PetscInt les_prt = 0;

PetscInt AL_Noslip=0;
PetscReal u_frame, v_frame, w_frame;
PetscInt MoveFrame = 0;

PetscInt ii_periodicWT=0, jj_periodicWT=0, kk_periodicWT=0; // periodic WT, a row/column of ghost wind turbines needs to be added
PetscReal Sx_WT=1.0, Sy_WT=1.0, Sz_WT=1.0;
PetscInt Nx_WT, Ny_WT, Nz_WT;

// Idealized water wave
PetscReal a_iww, lamda_iww, C_iww;

char path_inflow[256];

PetscReal prt_eps=1.e-20;

PetscInt New_wallmodel;

PetscInt tisteps;

// Sponge layer at outlet for levelset
//
PetscInt SpongeLayer=0;
PetscInt SpongeDistance=50;

PetscInt MoveCylinderTest=0;

PetscInt inflow_levelset=0;
PetscInt sublevel=0;
PetscReal smoothlevel=0;

// TUrbine model

PetscReal dfunc_wd=2.0;

PetscInt FixTipSpeedRatio=1;


// Mobile bed sediment

PetscReal dt_bed=1;  // for every dt_bed flow time steps, bed is computed once.
PetscReal particle_dens=1992.0;  // for every dt_bed flow time steps, bed is computed once.
PetscReal particle_D50=0.0018;  // for every dt_bed flow time steps, bed is computed once.
PetscInt smooth_bed=200;  // smooth coefficient for mobile bed.
PetscInt bed_avg=200;  // smooth coefficient for mobile bed.
PetscInt particlevel_model=0;  // model for parivle velocity.
PetscInt LS_bedConstr=1;  // using least squares for constructing xp, yp, zp
PetscReal Angle_repose=50;
PetscReal bed_porosity=0.5;
PetscReal sb_bed=0.01;
PetscInt number_smooth=2;
PetscReal deltab=0.01;
PetscReal rs_bed=6;

//
// end add (xiaolei)


// sediment begin
//

PetscInt sediment=0, input_ib_depth=0, conv_diff=0, SuspendedParticles=0, mobile_bed=0, projection_method=1, periodic_morpho=0, density_current=0;
PetscReal k_ss=0.0, w_s=0.0, Nseg=0.0, sed_density=20000, U_Bulk=0.0, d50=1000;
double inlet_sediment_flux=0.0;
PetscInt LiveBed=0;
PetscInt RigidBed=0;
PetscInt bed_roughness=0;

// sediment end
PetscBool	rstart_flg=PETSC_FALSE;
PetscBool	inlet_y_flag=PETSC_FALSE;
PetscBool	inlet_z_flag=PETSC_FALSE;

IBMNodes *ibm_ptr;
FSInfo *fsi_ptr;
UserCtx *user_ptr;




int file_exist(char *str)
{
  int r=0;

  if(!my_rank) {
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


PetscErrorCode Ucont_P_Binary_Input(UserCtx *user)
{
  PetscViewer	viewer;
  char filen[90];

  PetscInt N;
  VecGetSize(user->Ucont, &N);

  //MPI_Barrier(PETSC_COMM_WORLD);

  sprintf(filen, "%s/vfield%06d_%1d.dat", path, (int)ti, user->_this);
  PetscPrintf(PETSC_COMM_WORLD, "Reading %s ... \n", filen);

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerBinarySetMPIIO(viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, filen);
  /*
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
  */
  VecLoad(user->Ucont,viewer);
  PetscViewerDestroy(&viewer);

  //MPI_Barrier(PETSC_COMM_WORLD);

  //PetscOptionsClearValue("-vecload_block_size");

  PetscPrintf(PETSC_COMM_WORLD, "Reading pfield...\n");
  sprintf(filen, "%s/pfield%06d_%1d.dat", path, (int)ti, user->_this);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
  VecLoad(user->P,viewer);
  PetscViewerDestroy(&viewer);

  PetscPrintf(PETSC_COMM_WORLD, "Reading nvfield...\n");
  sprintf(filen, "%s/nvfield%06d_%1d.dat", path, (int)ti, user->_this);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
  VecLoad(user->Nvert_o,viewer);
  PetscViewerDestroy(&viewer);

  PetscPrintf(PETSC_COMM_WORLD, "Reading ufield...\n");
  sprintf(filen, "%s/ufield%06d_%1d.dat", path, (int)ti, user->_this);	// Seokkoo Kang
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
  VecLoad(user->Ucat,viewer);
  PetscViewerDestroy(&viewer);

  if(!immersed) {
    VecSet(user->Nvert, 0.);
    VecSet(user->Nvert_o, 0.);
  }

  DMGlobalToLocalBegin(user->da, user->Nvert_o, INSERT_VALUES, user->lNvert_o);
  DMGlobalToLocalEnd(user->da, user->Nvert_o, INSERT_VALUES, user->lNvert_o);

  DMGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
  DMGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);

  VecCopy(user->Ucont, user->Ucont_o);
  DMGlobalToLocalBegin(user->fda, user->Ucont_o, INSERT_VALUES, user->lUcont_o);
  DMGlobalToLocalEnd(user->fda, user->Ucont_o, INSERT_VALUES, user->lUcont_o);

  DMGlobalToLocalBegin(user->fda, user->Ucat, INSERT_VALUES, user->lUcat);
  DMGlobalToLocalEnd(user->fda, user->Ucat, INSERT_VALUES, user->lUcat);

  DMGlobalToLocalBegin(user->fda, user->Ucat, INSERT_VALUES, user->lUcat_old);
  DMGlobalToLocalEnd(user->fda, user->Ucat, INSERT_VALUES, user->lUcat_old);

  DMGlobalToLocalBegin(user->da, user->P, INSERT_VALUES, user->lP);
  DMGlobalToLocalEnd(user->da, user->P, INSERT_VALUES, user->lP);

  if(averaging) {	// Seokkoo Kang
    VecSet(user->Ucat_sum, 0);
    VecSet(user->Ucat_cross_sum, 0);
    VecSet(user->Ucat_square_sum, 0);
    VecSet(user->P_sum, 0);
    /*if(averaging>=2)*/ VecSet(user->P_square_sum, 0);
    if(levelset) {
      VecSet(user->Levelset_sum, 0);
      VecSet(user->Levelset_square_sum, 0);
    }

    if (conv_diff)
    {
      VecSet(user->Conc_sum, 0.);
      VecSet(user->Conc_cross_sum, 0.);
    }

    if(phase_averaging) {
      VecSet(user->Ucat_sum_phase, 0);
      VecSet(user->Ucat_cross_sum_phase, 0);
      VecSet(user->Ucat_square_sum_phase, 0);
      VecSet(user->P_sum_phase, 0);
      VecSet(user->P_square_sum_phase, 0);
    }
    /*
      if(les) {
      VecSet(user->Nut_sum, 0.);
      }*/

    if(rans) {
      VecSet(user->K_sum, 0.);
    }

    //if(averaging>=2) {
    //VecSet(user->P_cross_sum, 0);
    //}

    if(averaging>=3) {
      if(les) {
        //VecSet(user->tauS_sum, 0);
      }
      VecSet(user->Udp_sum, 0);
      VecSet(user->dU2_sum, 0);
      VecSet(user->UUU_sum, 0);
      VecSet(user->Vort_sum, 0);
      VecSet(user->Vort_square_sum, 0);
    }

    sprintf(filen, "%s/su0_%06d_%1d.dat", path, (int)ti, user->_this);
    FILE *fp=fopen(filen, "r");
    if(fp==NULL) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\n*** Cannot open %s, setting the statistical quantities to zero and contiues the computation ... ***\n\n", filen);
    }
    else {
      fclose(fp);
      MPI_Barrier(PETSC_COMM_WORLD);
      sprintf(filen, "%s/su0_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->Ucat_sum,viewer);
      PetscViewerDestroy(&viewer);

      PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

      sprintf(filen, "%s/su1_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->Ucat_cross_sum,viewer);
      PetscViewerDestroy(&viewer);

      PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

      sprintf(filen, "%s/su2_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->Ucat_square_sum,viewer);
      PetscViewerDestroy(&viewer);

      PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

      sprintf(filen, "%s/sp_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->P_sum,viewer);
      PetscViewerDestroy(&viewer);

      PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

      sprintf(filen, "%s/sp2_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->P_square_sum,viewer);
      PetscViewerDestroy(&viewer);

      PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

      if(phase_averaging) {
        int n, previous_ti;
        n_phase(&n, &previous_ti);

        if(n>0) {
          sprintf(filen, "%s/phase%03d_su0_%06d_%1d.dat", path, n, previous_ti, user->_this);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->Ucat_sum_phase,viewer);
          PetscViewerDestroy(&viewer);

          PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

          sprintf(filen, "%s/phase%03d_su1_%06d_%1d.dat", path, n, previous_ti, user->_this);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->Ucat_cross_sum_phase,viewer);
          PetscViewerDestroy(&viewer);

          PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

          sprintf(filen, "%s/phase%03d_su2_%06d_%1d.dat", path, n, previous_ti, user->_this);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->Ucat_square_sum_phase,viewer);
          PetscViewerDestroy(&viewer);

          PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

          sprintf(filen, "%s/phase%03d_sp_%06d_%1d.dat", path, n, previous_ti, user->_this);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->P_sum_phase,viewer);
          PetscViewerDestroy(&viewer);

          PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);

          sprintf(filen, "%s/phase%03d_sp2_%06d_%1d.dat", path, n, previous_ti, user->_this);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->P_square_sum_phase,viewer);
          PetscViewerDestroy(&viewer);

          PetscPrintf(PETSC_COMM_WORLD,"*** Read %s, continuing averaging ... ***\n", filen);
        }
        else {
          VecSet(user->Ucat_sum_phase, 0);
          VecSet(user->Ucat_cross_sum_phase, 0);
          VecSet(user->Ucat_square_sum_phase, 0);
          VecSet(user->P_square_sum_phase, 0);
        }
      }
      /*
        if(les) {
        sprintf(filen, "%s/snut_%06d_%1d.dat", path, (int)ti, user->_this);
        if( file_exist(filen) ) {
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoadIntoVector(viewer, user->Nut_sum);
        PetscViewerDestroy(&viewer);
        }
        }*/

      if(rans) {
        sprintf(filen, "%s/sk_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->K_sum,viewer);
        PetscViewerDestroy(&viewer);
      }

      if(levelset) {
        sprintf(filen, "%s/sl_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->Levelset_sum,viewer);
        PetscViewerDestroy(&viewer);

        sprintf(filen, "%s/sl2_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->Levelset_square_sum,viewer);
        PetscViewerDestroy(&viewer);
      }

      if (conv_diff)
      {
        sprintf(filen, "%s/sconc_%06d_%1d.dat", path, ti, user->_this);
        if (file_exist(filen))
        {
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->Conc_sum,viewer);
          PetscViewerDestroy(&viewer);
        }

        sprintf(filen, "%s/sconc_cross_sum_%06d_%1d.dat", path, ti, user->_this);
        if (file_exist(filen))
        {
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoad(user->Conc_cross_sum,viewer);
          PetscViewerDestroy(&viewer);
        }
      }

      if(averaging>=2) {

        /*
          sprintf(filen, "%s/sp1_%06d_%1d.dat", path, (int)ti, user->_this);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
          VecLoadIntoVector(viewer, user->P_cross_sum);
          PetscViewerDestroy(&viewer);
        */
      }

      if(averaging>=3) {
        if(les) {
          /*
            sprintf(filen, "%s/stauS_%06d_%1d.dat", path, (int)ti, user->_this);
            if( file_exist(filen) ) {
            PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
            VecLoadIntoVector(viewer, user->tauS_sum);
            PetscViewerDestroy(&viewer);
            }
            else PetscPrintf(PETSC_COMM_WORLD, "Cannot open %s !\n", filen);
          */
        }

        sprintf(filen, "%s/su3_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->Udp_sum,viewer);
        PetscViewerDestroy(&viewer);

        sprintf(filen, "%s/su4_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->dU2_sum,viewer);
        PetscViewerDestroy(&viewer);

        sprintf(filen, "%s/su5_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->UUU_sum,viewer);
        PetscViewerDestroy(&viewer);

        sprintf(filen, "%s/svo_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->Vort_sum,viewer);
        PetscViewerDestroy(&viewer);

        sprintf(filen, "%s/svo2_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->Vort_square_sum,viewer);
        PetscViewerDestroy(&viewer);
      }


    }
  }

  if(levelset) {

    sprintf(filen, "%s/lfield%06d_%1d.dat", path, (int)ti, user->_this);
    FILE *fp=fopen(filen, "r");

    if(fp==NULL) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\n*** Cannot open %s, terminates ... ***\n\n", filen);
      PetscFinalize();
      exit(1);
    }
    else {
      fclose(fp);

      MPI_Barrier(PETSC_COMM_WORLD);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->Levelset,viewer);
      PetscViewerDestroy(&viewer);

      DMGlobalToLocalBegin(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
      DMGlobalToLocalEnd(user->da, user->Levelset, INSERT_VALUES, user->lLevelset);
    }

  }

  if(les) {
    Vec Cs;
    VecDuplicate(user->P, &Cs);

    sprintf(filen, "%s/cs_%06d_%1d.dat", path, (int)ti, user->_this);
    FILE *fp=fopen(filen, "r");

    if(fp==NULL) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\n*** Cannot open %s, setting Cs to 0 and contiues the computation ... ***\n\n", filen);
      VecSet(Cs, 0);
    }
    else {
      fclose(fp);

      MPI_Barrier(PETSC_COMM_WORLD);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(Cs,viewer);
      PetscViewerDestroy(&viewer);
    }

    DMGlobalToLocalBegin(user->da, Cs, INSERT_VALUES, user->lCs);
    DMGlobalToLocalEnd(user->da, Cs, INSERT_VALUES, user->lCs);

    VecDestroy(&Cs);
  }

  if(rans) {
    // K-Omega
    sprintf(filen, "%s/kfield%06d_%1d.dat", path, (int)ti, user->_this);
    if( file_exist(filen) ) {
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->K_Omega,viewer);
      PetscViewerDestroy(&viewer);
    }
    else {
      //K_Omega_IC(user);
      PetscPrintf(PETSC_COMM_WORLD, "\nInitializing K-omega ... \n\n");
      K_Omega_IC(user);
      VecSet(user->lNu_t, user->ren);
    }

    VecCopy(user->K_Omega, user->K_Omega_o);

    DMGlobalToLocalBegin(user->fda2, user->K_Omega, INSERT_VALUES, user->lK_Omega);
    DMGlobalToLocalEnd(user->fda2, user->K_Omega, INSERT_VALUES, user->lK_Omega);

    DMGlobalToLocalBegin(user->fda2, user->K_Omega_o, INSERT_VALUES, user->lK_Omega_o);
    DMGlobalToLocalEnd(user->fda2, user->K_Omega_o, INSERT_VALUES, user->lK_Omega_o);

    if(rans==3) {
      // distance
      sprintf(filen, "%s/Distance_%1d.dat", path, user->_this);
      if( file_exist(filen) ) {
        PetscPrintf(PETSC_COMM_WORLD, "Reading %s !\n", filen);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(user->Distance,viewer);
        PetscViewerDestroy(&viewer);
      }
      else {
        PetscPrintf(PETSC_COMM_WORLD, "File %s does not exist. Recalculating distance function !\n", filen);
        /*Compute_Distance_Function(user);
          PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
          VecView(user->Distance, viewer);
          PetscViewerDestroy(&viewer);
        */
      }
    }
  }

  // xyang
  if(temperature) {
    // temperature
    sprintf(filen, "%s/tfield%06d_%1d.dat", path, ti, user->_this);
    if( file_exist(filen) ) {
      PetscPrintf(PETSC_COMM_WORLD, "Reading %s !\n", filen);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      VecLoad(user->Tmprt,viewer);
      PetscViewerDestroy(&viewer);
    }
    else {
      PetscPrintf(PETSC_COMM_WORLD, "\nInitializing temperature ... \n\n");
      Tmprt_IC(user);
    }

    if (les_prt) {
      Vec Pr_t;
      VecDuplicate(user->P, &Pr_t);

      sprintf(filen, "%s/prt%06d_%1d.dat", path, ti, user->_this);
      if( file_exist(filen) ) {
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
        VecLoad(Pr_t,viewer);
        PetscViewerDestroy(&viewer);
      }
      else {
        PetscPrintf(PETSC_COMM_WORLD, "\nInitializing Pr_t ... \n\n");
        VecSet(Pr_t, 0);
      }

      DMGlobalToLocalBegin(user->da, Pr_t, INSERT_VALUES, user->lPr_t);
      DMGlobalToLocalEnd(user->da, Pr_t, INSERT_VALUES, user->lPr_t);

      VecDestroy(&Pr_t);
    }

    VecCopy(user->Tmprt, user->Tmprt_o);

    DMGlobalToLocalBegin(user->da, user->Tmprt, INSERT_VALUES, user->lTmprt);
    DMGlobalToLocalEnd(user->da, user->Tmprt, INSERT_VALUES, user->lTmprt);

    DMGlobalToLocalBegin(user->da, user->Tmprt_o, INSERT_VALUES, user->lTmprt_o);
    DMGlobalToLocalEnd(user->da, user->Tmprt_o, INSERT_VALUES, user->lTmprt_o);

  }
  //
  if(conv_diff)
  {
    sprintf(filen, "%s/cfield%06d_%1d.dat", path, ti, user->_this);
    if (file_exist(filen))
    {
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
      PetscPrintf(PETSC_COMM_WORLD, "Reading %s !\n", filen);
      VecLoad(user->Conc,viewer);
      PetscViewerDestroy(&viewer);
    }

    else
    {
      PetscPrintf(PETSC_COMM_WORLD,"\n\n Cannot open %s, terminates ... So Will Do Initialization...\n\n", filen);
      PetscPrintf(PETSC_COMM_WORLD, "\nInitializing Concentration eqn ......... \n\n");
      Conv_Diff_IC(user);
    }

    VecCopy (user->Conc, user->Conc_o);
    DMGlobalToLocalBegin(user->da, user->Conc, INSERT_VALUES, user->lConc);
    DMGlobalToLocalEnd(user->da, user->Conc, INSERT_VALUES, user->lConc);

    DMGlobalToLocalBegin(user->da, user->Conc_o, INSERT_VALUES, user->lConc_o);
    DMGlobalToLocalEnd(user->da, user->Conc_o, INSERT_VALUES, user->lConc_o);
  }



  PetscPrintf(PETSC_COMM_WORLD, "Finished reading ...\n");
  return 0;
}

PetscErrorCode Ucat_Binary_Output(UserCtx *user)
{
  PetscViewer	viewer;
  char filen[80];

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  MPI_Barrier(PETSC_COMM_WORLD);

  sprintf(filen, "%s/ufield%06d_%1d.dat", path, (int)ti, user->_this);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
  VecView(user->Ucat, viewer);
  PetscViewerDestroy(&viewer);
  sprintf(filen, "%s/ufield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);

  MPI_Barrier(PETSC_COMM_WORLD);

  return 0;
}


void write_output_file (UserCtx *user, Vec V, char *file)
{
  PetscViewer viewer;
  char s[256];

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerBinarySetMPIIO(viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, file);
  VecView(V, viewer);
  PetscViewerDestroy(&viewer);

  if(my_rank) {
    sprintf(s, "%s.info", file);
    unlink(s);
  }
}

PetscErrorCode Phase_Averaging_Output(UserCtx *user)
{
  int n, previous_ti;
  n_phase(&n, &previous_ti);

  if(n>0) {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscViewer	viewer;
    char filen[80];

    sprintf(filen, "%s/phase%03d_su0_%06d_%1d.dat", path, n, previous_ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat_sum_phase, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/phase%03d_su0_%06d_%1d.dat.info", path, n, previous_ti, user->_this);
    if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/phase%03d_su1_%06d_%1d.dat", path, n, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat_cross_sum_phase, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/phase%03d_su1_%06d_%1d.dat.info", path, n, previous_ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/phase%03d_su2_%06d_%1d.dat", path, n, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat_square_sum_phase, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/phase%03d_su2_%06d_%1d.dat.info", path, n, previous_ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/phase%03d_sp_%06d_%1d.dat", path, n, previous_ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->P_sum_phase, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/phase%03d_sp_%06d_%1d.dat.info", path, n, previous_ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/phase%03d_sp2_%06d_%1d.dat", path, n, previous_ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->P_square_sum_phase, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/phase%03d_sp2_%06d_%1d.dat.info", path, n, previous_ti, user->_this);
    if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/nvfield%06d_%1d.dat", path, previous_ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Nvert, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/nvfield%06d_%1d.dat.info", path, previous_ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);
  }

  return 0;
}

int delete_count=0;

PetscErrorCode Ucont_P_Binary_Output(UserCtx *user)
{
  PetscViewer	viewer;
  char filen[80];

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


  MPI_Barrier(PETSC_COMM_WORLD);

  sprintf(filen, "%s/vfield%06d_%1d.dat", path, (int)ti, user->_this);
  write_output_file (user, user->Ucont, filen);
  /*
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucont, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/vfield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
  */
  MPI_Barrier(PETSC_COMM_WORLD);

  sprintf(filen, "%s/ufield%06d_%1d.dat", path, (int)ti, user->_this);
  write_output_file (user, user->Ucat, filen);
  /*
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/ufield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
  */
  MPI_Barrier(PETSC_COMM_WORLD);

  sprintf(filen, "%s/pfield%06d_%1d.dat", path, (int)ti, user->_this);
  write_output_file (user, user->P, filen);
  /*
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->P, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/pfield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
  */
  MPI_Barrier(PETSC_COMM_WORLD);

  if(qcr) {
    Vec Q;
    VecDuplicate(user->P, &Q);
    Compute_Q(user,  Q);

    sprintf(filen, "%s/qfield%06d_%1d.dat", path, (int)ti, user->_this);
    write_output_file (user, Q, filen);
    /*
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(Q, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/qfield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
    */
    VecDestroy(&Q);
    MPI_Barrier(PETSC_COMM_WORLD);
  }

  sprintf(filen, "%s/nvfield%06d_%1d.dat", path, (int)ti, user->_this);
  write_output_file (user, user->Nvert, filen);
  /*
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Nvert, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/nvfield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
  */

  MPI_Barrier(PETSC_COMM_WORLD);

  if(averaging && ti!=0) {	// Seokkoo Kang
    sprintf(filen, "%s/su0_%06d_%1d.dat", path, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat_sum, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/su0_%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/su1_%06d_%1d.dat", path, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat_cross_sum, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/su1_%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/su2_%06d_%1d.dat", path, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->Ucat_square_sum, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/su2_%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/sp_%06d_%1d.dat",path, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->P_sum, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/sp_%06d_%1d.dat.info",path, (int)ti, user->_this);	if(!rank) unlink(filen);

    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(filen, "%s/sp2_%06d_%1d.dat",path, (int)ti, user->_this);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(user->P_square_sum, viewer);
    PetscViewerDestroy(&viewer);
    sprintf(filen, "%s/sp2_%06d_%1d.dat.info",path, (int)ti, user->_this);
    if(!rank) unlink(filen);
    /*
      if(les) {
      sprintf(filen, "%s/snut_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Nut_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/snut_%06d_%1d.dat.info",path, (int)ti, user->_this);
      if(!rank) unlink(filen);
      }*/

    if(rans) {
      sprintf(filen, "%s/sk_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->K_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/sk_%06d_%1d.dat.info", path, (int)ti, user->_this);
      if(!rank) unlink(filen);
    }

    if(levelset) {
      sprintf(filen, "%s/sl_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Levelset_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/sl_%06d_%1d.dat.info",path, (int)ti, user->_this);
      if(!rank) unlink(filen);

      sprintf(filen, "%s/sl2_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Levelset_square_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/sl2_%06d_%1d.dat.info",path, (int)ti, user->_this);
      if(!rank) unlink(filen);
    }

    if(temperature) {
      sprintf(filen, "%s/st_%06d_%1d.dat",path, ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->T_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/st_%06d_%1d.dat.info",path, ti, user->_this);
      if(!rank) unlink(filen);

      sprintf(filen, "%s/st2_%06d_%1d.dat",path, ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->TT_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/st2_%06d_%1d.dat.info",path, ti, user->_this);
      if(!rank) unlink(filen);


      sprintf(filen, "%s/stu_%06d_%1d.dat",path, ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->TU_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/stu_%06d_%1d.dat.info",path, ti, user->_this);
      if(!rank) unlink(filen);

      sprintf(filen, "%s/stp_%06d_%1d.dat",path, ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->TP_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/stp_%06d_%1d.dat.info",path, ti, user->_this);
      if(!rank) unlink(filen);


      if (les_prt) {
        sprintf(filen, "%s/sprt_%06d_%1d.dat",path, ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
        VecView(user->Prt_sum, viewer);
        PetscViewerDestroy(&viewer);
        sprintf(filen, "%s/sprt_%06d_%1d.dat.info",path, ti, user->_this);
        if(!rank) unlink(filen);

      }
      //
    }

    if (conv_diff)
    {
      sprintf(filen, "%s/sconc_%06d_%1d.dat",path, ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Conc_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/sconc_%06d_%1d.dat.info",path, ti, user->_this);
      if(!rank)
        unlink(filen);

      sprintf(filen, "%s/sconc_cross_sum_%06d_%1d.dat",path, ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Conc_cross_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/sconc_cross_sum_%06d_%1d.dat.info",path, ti, user->_this);
      if(!rank)
        unlink(filen);

    }

    if(averaging>=2) {

      /*
        sprintf(filen, "%s/sp1_%06d_%1d.dat",path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
        VecView(user->P_cross_sum, viewer);
        PetscViewerDestroy(&viewer);
        sprintf(filen, "%s/sp1_%06d_%1d.dat.info",path, (int)ti, user->_this);
        if(!rank) unlink(filen);
      */
    }

    if(averaging>=3) {
      /*
        if(les) {
        sprintf(filen, "%s/stauS_%06d_%1d.dat", path, (int)ti, user->_this);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
        VecView(user->tauS_sum, viewer);
        PetscViewerDestroy(&viewer);
        sprintf(filen, "%s/stauS_%06d_%1d.dat.info",path, (int)ti, user->_this);if(!rank) unlink(filen);
        }*/

      sprintf(filen, "%s/su3_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Udp_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/su3_%06d_%1d.dat.info", path, (int)ti, user->_this);if(!rank) unlink(filen);

      sprintf(filen, "%s/su4_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->dU2_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/su4_%06d_%1d.dat.info", path, (int)ti, user->_this);if(!rank) unlink(filen);

      sprintf(filen, "%s/su5_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->UUU_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/su5_%06d_%1d.dat.info",path, (int)ti, user->_this);if(!rank) unlink(filen);

      sprintf(filen, "%s/svo_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Vort_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/svo_%06d_%1d.dat.info",path, (int)ti, user->_this);
      if(!rank) unlink(filen);

      sprintf(filen, "%s/svo2_%06d_%1d.dat",path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Vort_square_sum, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/svo2_%06d_%1d.dat.info",path, (int)ti, user->_this);
      if(!rank) unlink(filen);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }

  if(levelset) {
    sprintf(filen, "%s/lfield%06d_%1d.dat", path, (int)ti, user->_this);
    write_output_file (user, user->Levelset, filen);
    /*
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Levelset, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/lfield%06d_%1d.dat.info",path, (int)ti, user->_this);	if(!rank) unlink(filen);
    */
  }

  if(les) {
    Vec Cs;

    VecDuplicate(user->P, &Cs);
    DMLocalToGlobalBegin(user->da, user->lCs, INSERT_VALUES, Cs);
    DMLocalToGlobalEnd(user->da, user->lCs, INSERT_VALUES, Cs);

    sprintf(filen, "%s/cs_%06d_%1d.dat", path, (int)ti, user->_this);
    write_output_file (user, Cs, filen);
    /*
      sprintf(filen, "%s/cs_%06d_%1d.dat", path, (int)ti, user->_this);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(Cs, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/cs_%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
    */
    MPI_Barrier(PETSC_COMM_WORLD);
    VecDestroy(&Cs);
  }

  if(rans) {
    sprintf(filen, "%s/kfield%06d_%1d.dat", path, (int)ti, user->_this);
    write_output_file (user, user->K_Omega, filen);
    /*
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->K_Omega, viewer);
      PetscViewerDestroy(&viewer);
      sprintf(filen, "%s/kfield%06d_%1d.dat.info", path, (int)ti, user->_this);	if(!rank) unlink(filen);
    */
    MPI_Barrier(PETSC_COMM_WORLD);
  }

  // xyang
  if(temperature) {
    sprintf(filen, "%s/tfield%06d_%1d.dat", path, ti, user->_this);
    write_output_file (user, user->Tmprt, filen);
    /*
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
      VecView(user->Tmprt, viewer);
      PetscViewerDestroy(viewer);
      sprintf(filen, "%s/tfield%06d_%1d.dat.info", path, ti, user->_this);
      if(!rank) unlink(filen);
    */
    PetscBarrier(PETSC_NULL);

    if (les_prt) {

      Vec Pr_t;

      VecDuplicate(user->P, &Pr_t);
      DMLocalToGlobalBegin(user->da, user->lPr_t, INSERT_VALUES, Pr_t);
      DMLocalToGlobalEnd(user->da, user->lPr_t, INSERT_VALUES, Pr_t);

      sprintf(filen, "%s/prt%06d_%1d.dat", path, ti, user->_this);
      write_output_file (user, Pr_t, filen);
      /*
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
        VecView(user->Pr_t, viewer);
        PetscViewerDestroy(viewer);
        sprintf(filen, "%s/prt%06d_%1d.dat.info", path, ti, user->_this);
        if(!rank) unlink(filen);
      */
      PetscBarrier(PETSC_NULL);
      VecDestroy(&Pr_t);
    }
  }
  //

  if (conv_diff)
  {
    sprintf(filen, "%s/cfield%06d_%1d.dat", path, ti, user->_this);
    write_output_file (user, user->Conc, filen);

    PetscBarrier(PETSC_NULL);

    //PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    //VecView(user->Conc, viewer);
    //PetscViewerDestroy(&viewer);
    //sprintf(filen, "%s/cfield%06d_%1d.dat.info",path, ti, user->_this);	if(!rank) unlink(filen);
  }


  if(!rank && delete_previous_file && delete_count++>=2 && ti-tiout*2!=0) {
    sprintf(filen, "%s/vfield%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);

    if(!(tiout_ufield>0 && ti == (ti/tiout_ufield) * tiout_ufield && ti<=tiend_ufield)) {
      sprintf(filen, "%s/ufield%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
    }
    sprintf(filen, "%s/pfield%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
    sprintf(filen, "%s/nvfield%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
    if(averaging) {
      sprintf(filen, "%s/sp_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      sprintf(filen, "%s/su0_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      sprintf(filen, "%s/su1_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      sprintf(filen, "%s/su2_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      sprintf(filen, "%s/sp2_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);

      if(levelset) {
        sprintf(filen, "%s/sl_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
        sprintf(filen, "%s/sl2_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
        if(!rank) unlink(filen);
      }

      if(rans) {
        sprintf(filen, "%s/sk_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      }

      if(averaging>=2) {
        //sprintf(filen, "%s/sp1_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      }
      if(averaging>=3) {
        sprintf(filen, "%s/su3_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
        if(!rank) unlink(filen);
        sprintf(filen, "%s/su4_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
        if(!rank) unlink(filen);
        sprintf(filen, "%s/su5_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
        if(!rank) unlink(filen);

        sprintf(filen, "%s/svo_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
        sprintf(filen, "%s/svo2_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
      }
      if(les) {
        sprintf(filen, "%s/stauS_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
        if(!rank) unlink(filen);
        sprintf(filen, "%s/snut_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
        if(!rank) unlink(filen);
      }
    }
    if(les>=2) {
      sprintf(filen, "%s/cs_%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
    }
    if(rans) {
      sprintf(filen, "%s/kfield%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);	if(!rank) unlink(filen);
    }
    if(levelset) {
      sprintf(filen, "%s/lfield%06d_%1d.dat", path, (int)(ti-tiout*2), user->_this);
      if(!rank) unlink(filen);
    }

  }
  MPI_Barrier(PETSC_COMM_WORLD);

  return 0;
}

PetscErrorCode Divergence(UserCtx *user)
{
  DM		da = user->da, fda = user->fda;
  DMDALocalInfo	info = user->info;

  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;

  PetscInt	lxs, lys, lzs, lxe, lye, lze;
  PetscInt	i, j, k;

  Vec		Div;
  PetscReal	***div, ***aj, ***nvert;
  Cmpnts	***ucont;
  PetscReal	maxdiv;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  DMDAVecGetArray(fda,user->lUcont, &ucont);
  DMDAVecGetArray(da, user->lAj, &aj);
  VecDuplicate(user->P, &Div);
  DMDAVecGetArray(da, Div, &div);
  DMDAVecGetArray(da, user->lNvert, &nvert);
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {

	maxdiv = fabs((ucont[k][j][i].x - ucont[k][j][i-1].x + ucont[k][j][i].y - ucont[k][j-1][i].y + ucont[k][j][i].z - ucont[k-1][j][i].z)*aj[k][j][i]);
	//if(i==mx-2) printf("%f %f %f %f %f %f\n", ucont[k][j][i].x, ucont[k][j][i-1].x, ucont[k][j][i].y, ucont[k][j-1][i].y, ucont[k][j][i].z, ucont[k-1][j][i].z);
	if (nvert[k][j][i] + nvert[k+1][j][i] + nvert[k-1][j][i] + nvert[k][j+1][i] + nvert[k][j-1][i] + nvert[k][j][i+1] + nvert[k][j][i-1] > 0.1) maxdiv = 0.;



	div[k][j][i] = maxdiv;

      }
    }
  }

  if (zs==0) {
    k=0;
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	div[k][j][i] = 0.;
      }
    }
  }

  if (ze == mz) {
    k=mz-1;
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	div[k][j][i] = 0.;
      }
    }
  }

  if (xs==0) {
    i=0;
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	div[k][j][i] = 0.;
      }
    }
  }

  if (xe==mx) {
    i=mx-1;
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	div[k][j][i] = 0;
      }
    }
  }

  if (ys==0) {
    j=0;
    for (k=zs; k<ze; k++) {
      for (i=xs; i<xe; i++) {
	div[k][j][i] = 0.;
      }
    }
  }

  if (ye==my) {
    j=my-1;
    for (k=zs; k<ze; k++) {
      for (i=xs; i<xe; i++) {
	div[k][j][i] = 0.;
      }
    }
  }
  DMDAVecRestoreArray(da, Div, &div);
  VecMax(Div, &i, &maxdiv);
  PetscPrintf(PETSC_COMM_WORLD, "Maxdiv %d %d %e\n", (int)ti, i, maxdiv);
  int mi;


  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (mi=xs; mi<xe; mi++) {
	if (lidx(mi,j,k,user) ==i) {
	  PetscPrintf(PETSC_COMM_SELF, "MMa %d %d %d\n", mi,j, k);
	}
      }
    }
  }



  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (!rank) {
    FILE *f;
    char filen[80];
    sprintf(filen, "%s/Converge_dU", path);
    f = fopen(filen, "a");
    PetscFPrintf(PETSC_COMM_WORLD, f, " Maxdiv=%.2e\n", maxdiv);
    fclose(f);
  }

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(fda, user->lUcont, &ucont);
  DMDAVecRestoreArray(da, user->lAj, &aj);


  /*
    char fname[80];
    sprintf(fname,"DIV");
    TECIOOut_rhs_da(user, Div, fname);
  */


  VecDestroy(&Div);
  return(0);
}

void write_data(UserCtx *user)
{
  DM		da = user->da, fda = user->fda;
  DMDALocalInfo	info = user->info;

  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;

  int	lxs, lys, lzs, lxe, lye, lze;
  int	i, j, k;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  double lvol=0, vol=0;

  PetscReal ***p, ***aj, ***nvert, ***rho, ***level;
  Cmpnts	***ucat, ***ucont, ***csi, ***eta, ***zet, ***cent;

  if(levelset) {
    DMDAVecGetArray(da, user->lDensity, &rho);
    DMDAVecGetArray(da, user->lLevelset, &level);
  }
  DMDAVecGetArray(da,user->lAj, &aj);
  DMDAVecGetArray(da, user->lP, &p);
  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(fda,user->lCsi, &csi);
  DMDAVecGetArray(fda,user->lEta, &eta);
  DMDAVecGetArray(fda,user->lZet, &zet);
  DMDAVecGetArray(fda,user->lUcat, &ucat);
  DMDAVecGetArray(fda,user->lUcont, &ucont);
  DMDAVecGetArray(fda,user->lCent, &cent);



  if (ti==tistart && nsave_points>0) {

    int *iclose, *jclose, *kclose;
    int *sum_iclose, *sum_jclose, *sum_kclose;

    iclose= (int *) malloc(nsave_points*sizeof(int));
    jclose= (int *) malloc(nsave_points*sizeof(int));
    kclose= (int *) malloc(nsave_points*sizeof(int));

    sum_iclose= (int *) malloc(nsave_points*sizeof(int));
    sum_jclose= (int *) malloc(nsave_points*sizeof(int));
    sum_kclose= (int *) malloc(nsave_points*sizeof(int));


    PetscPrintf(PETSC_COMM_WORLD, "save %d flow points\n\n", nsave_points);
    int count[nsave_points], sum_count[nsave_points];
    for(int m=0; m<nsave_points; m++) {
      double XX=save_coor[m][0];
      double YY=save_coor[m][1];
      double ZZ=save_coor[m][2];

      double dis, dis_min;
      dis_min=1.e9;

      for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
          for (i=xs; i<xe; i++) {
            dis=pow(XX-cent[k][j][i].x,2)+pow(YY-cent[k][j][i].y,2)+pow(ZZ-cent[k][j][i].z,2);

            if (dis<dis_min) {
              dis_min=dis;
              iclose[m]=i;
              jclose[m]=j;
              kclose[m]=k;
            }
          }



      double dmin_global;
      MPI_Allreduce (&dis_min, &dmin_global, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);
      double diff=fabs(dis_min-dmin_global);
      if (diff<1.e-4) {
        count[m]=1;
      }
      else {
        count[m]=0;
        iclose[m]=0; jclose[m]=0; kclose[m]=0;
      }

    }

    PetscBarrier(PETSC_NULL);

    MPI_Allreduce (&iclose[0], &sum_iclose[0], nsave_points, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce (&jclose[0], &sum_jclose[0], nsave_points, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce (&kclose[0], &sum_kclose[0], nsave_points, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce (&count[0], &sum_count[0], nsave_points, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);


    for (int m=0; m<nsave_points; m++) {
      save_point[m][0]=sum_iclose[m]/sum_count[m];
      save_point[m][1]=sum_jclose[m]/sum_count[m];
      save_point[m][2]=sum_kclose[m]/sum_count[m];
      PetscPrintf(PETSC_COMM_WORLD, "save flow points at x=%le y=%le z=%le\n", save_coor[m][0],  save_coor[m][1],  save_coor[m][2] );
      PetscPrintf(PETSC_COMM_WORLD, "save flow points at i=%d j=%d k=%d\n", save_point[m][0],  save_point[m][1],  save_point[m][2] );
    }


    free(iclose);
    free(jclose);
    free(kclose);

    free(sum_iclose);
    free(sum_jclose);
    free(sum_kclose);

  }




  // for sloshing recording
  int ci=mx/2, ck=mz/2;//mz should be even //(mz-3)/2 + 1;
  double lz_sloshing=-10, z_sloshing;

  std::vector<double> l_freesurfacez ( mx * mz ), freesurfacez ( mx * mz );
  std::vector<double> l_xcoord ( mx * mz ), xcoord ( mx * mz );
  std::vector<double> l_ycoord ( mx * mz ), ycoord ( mx * mz );

  if(levelset) {
    for(i=0; i<mx; i++)
      for(k=0; k<mz; k++) {
        l_freesurfacez[k*mx+i] = -1000;
        l_xcoord[k*mx+i] = -1000;
        l_ycoord[k*mx+i] = -1000;
      }
  }

  for (k=lzs; k<lze; k++)
    for (j=lys; j<lye; j++)
      for (i=lxs; i<lxe; i++) {
        if(nvert[k][j][i]>0.1) continue;
        if(levelset) {
          //if(levelset==2) printf("check main.c - kang !!!!\n");
          lvol += rho[k][j][i] / aj[k][j][i];
          if ( i==ci && k==ck) { // center
            if( level[k][j][i]>=0 && level[k][j+1][i]<0 ) {	// water surface is above my cell center
              lz_sloshing = cent[k][j][i].z + level[k][j][i];
            }
            /*
              else if( level[k][j][i]<0 && level[k][j-1][i]>=0 ) {	// water surface is below my cell center
              lz_sloshing = cent[k][j][i].z + level[k][j][i];
              }*/
          }

          if( level[k][j][i]>=0 && level[k][j+1][i]<0 ) {	// water surface is above my cell center
            l_freesurfacez[k*mx+i] = cent[k][j][i].z + level[k][j][i];
            l_xcoord[k*mx+i] = cent[k][j][i].x;
            l_ycoord[k*mx+i] = cent[k][j][i].y;
          }
          /*
            else if( level[k][j][i]<0 && level[k][j-1][i]>=0 ) {	// water surface is below my cell center
            l_freesurfacez[k*mx+i] = cent[k][j][i].z + level[k][j][i];
            l_xcoord[k*mx+i] = cent[k][j][i].x;
            }*/
        }

        for(int m=0; m<nsave_points; m++) {
          if(i==save_point[m][0] && j==save_point[m][1] && k==save_point[m][2]) {
            double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
            double dpdc, dpde, dpdz;
            double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
            double dp_dx, dp_dy, dp_dz;

            double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
            double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
            double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
            double ajc = aj[k][j][i];

            double Ai = sqrt ( csi0*csi0 + csi1*csi1 + csi2*csi2 );
            double Aj = sqrt ( eta0*eta0 + eta1*eta1 + eta2*eta2 );
            double Ak = sqrt ( zet0*zet0 + zet1*zet1 + zet2*zet2 );

            double U = 0.5*(ucont[k][j][i].x+ucont[k][j][i-1].x) / Ai;
            double V = 0.5*(ucont[k][j][i].y+ucont[k][j-1][i].y) / Aj;
            double W = 0.5*(ucont[k][j][i].z+ucont[k-1][j][i].z) / Ak;

            Compute_du_center ( i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
            Compute_du_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

            Compute_dscalar_center ( i, j, k, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz );
            Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz );

            double vort_x = dw_dy - dv_dz,  vort_y = du_dz - dw_dx, vort_z = dv_dx - du_dy;

            FILE *f;
            char filen[80];

            sprintf(filen, "%s/Flow0_%.2e_%.2e_%.2e_dt_%g.dat", path, save_coor[m][0], save_coor[m][1], save_coor[m][2], user->dt);
            f = fopen(filen, "a");
            //if(ti==tistart) fprintf(f, "\n");//fprintf(f, "\n\n**** Beginning to log time history of u v w p U V W dudx dudy dudz dvdx dvdy dvdz dwdx dwdy dwdz at point (%d,%d,%d) ****\n", i, j, k);
            fprintf(f, "%d %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e \n", (int)ti, cent[k][j][i].x, cent[k][j][i].y, cent[k][j][i].z, ucat[k][j][i].x, ucat[k][j][i].y, ucat[k][j][i].z, p[k][j][i], U, V, W, vort_x, vort_y, vort_z);
            fclose(f);

            sprintf(filen, "%s/Flow1_%.2ed_%.2e_%.2e_dt_%g.dat", path, save_coor[m][0], save_coor[m][1], save_coor[m][2], user->dt);
            f = fopen(filen, "a");
            //if(ti==tistart) fprintf(f, "\n");
            fprintf(f, "%d %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e\n", (int)ti, cent[k][j][i].x, cent[k][j][i].y, cent[k][j][i].z, du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz, dp_dx, dp_dy, dp_dz);
            fclose(f);

            break;
          }
        }
      }

  if(levelset) {
    GlobalSum_Root(&lvol, &vol, PETSC_COMM_WORLD);
    GlobalMax_Root(&lz_sloshing, &z_sloshing, PETSC_COMM_WORLD);
    MPI_Reduce(&l_freesurfacez[0], &freesurfacez[0], mx*mz, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&l_xcoord[0], &xcoord[0], mx*mz, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&l_ycoord[0], &ycoord[0], mx*mz, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);

    if(!my_rank) {
      double _time = (ti+1)*user->dt;
      double a_sloshing_exact;
      double err_sum=0;
      int N = 0;

      //for(i=1; i<mx-1; i++)
      i = ci; // for 1D sloshing
      for(k=1; k<mz-1; k++) { // exact solution for linear sloshing
        double a = sloshing_a, b = sloshing_b, d = sloshing_d, g = 1.0; // 0.05
        double k2 = 2*M_PI/b, w2 = sqrt ( k2 * g * tanh (k2*d) ), k4 = 4*M_PI/b, w4 = sqrt ( k4 * g * tanh (k4*d) );
        double x = xcoord [k*mx+i];
        double y = ycoord [k*mx+i];
        double exact=0;

        if(inviscid && sloshing==1) {
          exact = a * cos(w2*_time) * cos(k2*x);
          exact += 0.125/g * ( 2*pow(w2*a,2.) * cos(2.*w2*_time) + pow(a/w2,2.) * ( pow(k2*g,2.) + pow(w2,4.) ) - pow(a/w2,2.) * ( pow(k2*g,2.) + 3.*pow(w2,4.) ) * cos(w4*_time) );
        }
        else if(sloshing==1) {
          double nu = mu_water/rho_water;
          double eta0 = a * cos(k2*x);
          exact = eta0;
          exact -= eta0 /(1.+4.*nu*nu*k2*k2*k2/g) * ( 1. - exp(-2.*nu*k2*k2*_time) * ( cos(sqrt(k2*g)*_time) + 2.*nu*k2*k2*sin(sqrt(k2*g)*_time)/sqrt(k2*g)) );
        }
        else if(sloshing==2 && i==ci && k==ck) {
          double L = 20., d=1., Beta=0.25, g=fabs(gravity_z), a=0.1; // L = width of the 3D tank
          printf("x=%f, y=%f\n", x, y );
          double eta0 = a * exp ( -Beta * ( pow(x-L/2, 2) + pow(y-L/2, 2) ) );
          std::complex<double> I(0,1);
          std::complex<double> one(1,0);
          std::complex<double> val(0,0);
          for(int m=0; m<40; m++)
            for(int n=0; n<40; n++) {
              double coeff_m=2., coeff_n=2.;
              double k_mn = sqrt( pow(M_PI/L,2.) * (m*m + n*n) );
              double omega_mn = sqrt ( g * k_mn * tanh (k_mn * d) );
              if(m==0) coeff_m=1.;
              if(n==0) coeff_n=1.;
              std::complex<double> Integral;
              Integral   = M_PI * a / (16*Beta) * exp ( -0.25*M_PI * ( 2.*(m+n)*I + (m*m+n*n)*M_PI/(Beta*L*L)*one ) );
              Integral *= ( one + exp(m*M_PI*I) ) * ( one + exp(n*M_PI*I) );
              Integral *= ERF ( (Beta*L*L - m*M_PI*I) * pow(2*sqrt(Beta)*L, -1) ) + ERF ( (Beta*L*L + m*M_PI*I) * pow(2*sqrt(Beta)*L, -1) );
              Integral *= ERF ( (Beta*L*L - n*M_PI*I) * pow(2*sqrt(Beta)*L, -1) ) + ERF ( (Beta*L*L + n*M_PI*I) * pow(2*sqrt(Beta)*L, -1) );

              double eta_mn = (1./(L*L)) * coeff_m * coeff_n * Integral.real();
              val += eta_mn * exp ( -I * omega_mn * _time ) * cos (n * M_PI / L * x) * cos (m * M_PI / L * y);
            }
          exact = val.real();
        }
        /*
          std::complex<double> val(1.,0), one(1.,0);
          double kappa = g/nu/nu/k2/k2/k2;

          // double g1R = -1.651678082, g1I = 2.081702511, g2R = 1.651678082, g2I = 1.767086763; // for nu=0.01, d=1,b=1
          double g1R = -5.590248751, g1I = 5.694713691;
          double g2R = 5.590248751, g2I = 5.663214475; // for nu=0.001, d=1,b=1
          double delta = 4 * (6*g1R*g1R + 1) * (2*g1R*g1R + 1) - 4 * (kappa+1);
          double A1R = -2/delta * g1R, A2R = 2/delta * g1R;
          double A1I = -(2*pow(g1R,3.)+1.)/(g1R*g1I*delta), A2I = -(2*pow(g1R,3.)-1.)/(g1R*g2I*delta);

          std::complex<double> g1(g1R, g1I), g2(g2R, g2I);
          std::complex<double> A1(A1R, A1I), A2(A2R, A2I);

          std::complex<double> erf1 = ERF(g1*k2*sqrt(nu*_time)); // erf(g1*k2*sqrt(nu*t))
          std::complex<double> erf2 = ERF(g2*k2*sqrt(nu*_time));

          val =  (A1/(one-g1*g1)) * (-g1*exp( (-one+g1*g1)*nu*k2*k2*_time ) * (one+erf1) + g1 + std::complex<double>(erf(k2*sqrt(nu*_time)),0) );
          val += (A2/(one-g2*g2)) * (-g2*exp( (-one+g2*g2)*nu*k2*k2*_time ) * (one+erf2) + g2 + std::complex<double>(erf(k2*sqrt(nu*_time)),0) );
          //val *= eta0;

          exact = eta0 * ( 1 - 2.0 * kappa *  val.real() );
        */

        double computed = freesurfacez [k*mx+i] -1.0;

        err_sum += pow( computed - exact, 2.0 );
        N ++;

        if ( i==ci && k==ck) { // center
          //x = b * 0.5;
          a_sloshing_exact = exact;//a * cos(w2*_time) * cos(k2*x);
        }
      }

      err_sum = sqrt ( err_sum ) / (double)N;

      char filen[256];
      sprintf(filen, "%s/mass.dat", path);
      FILE *fp = fopen(filen, "a");
      fprintf(fp, "%d %.10e\n", (int)ti, vol );
      fclose(fp);

      /*
        double a = 0.05, b = 2.0, d = 1.0, g = 1., x = 1.0;
        double k2 = 2*M_PI/b, k4 = 4*M_PI/b;
        double w2 = sqrt ( k2 * g * tanh (k2*d) ), w4 = sqrt ( k4 * g * tanh (k4*d) );

        z_sloshing_exact = a * cos(w2*_time) * cos(k2*x);
        z_sloshing_exact  += 0.125/g * ( 2*pow(w2*a,2.) * cos(2.*w2*_time) + pow(a/w2,2.) * ( pow(k2*g,2.) + pow(w2,4.) ) - pow(a/w2,2.) * ( pow(k2*g,2.) + 3.*pow(w2,4.) ) * cos(w4*_time) );
      */

      sprintf(filen, "%s/sloshing.dat", path);
      fp = fopen(filen, "a");
      fprintf(fp, "%e %.7e %.7e %.7e\n", _time, z_sloshing-1, a_sloshing_exact, err_sum);
      fclose(fp);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
  }

  if(user->bctype[0]==11 && ti) {
    GlobalSum_Root(&user->lA_cyl, &user->A_cyl, PETSC_COMM_WORLD);
    GlobalSum_Root(&user->lA_cyl_x, &user->A_cyl_x, PETSC_COMM_WORLD);
    GlobalSum_Root(&user->lA_cyl_z, &user->A_cyl_z, PETSC_COMM_WORLD);

    GlobalSum_Root(&user->lFpx_cyl, &user->Fpx_cyl, PETSC_COMM_WORLD);
    GlobalSum_Root(&user->lFpz_cyl, &user->Fpz_cyl, PETSC_COMM_WORLD);
    GlobalSum_Root(&user->lFvx_cyl, &user->Fvx_cyl, PETSC_COMM_WORLD);
    GlobalSum_Root(&user->lFvz_cyl, &user->Fvz_cyl, PETSC_COMM_WORLD);

    //double Cpx=user->Fpx_cyl/user->A_cyl*2., Cpz=user->Fpz_cyl/user->A_cyl*2.;
    //double Cvx=user->Fvx_cyl/user->A_cyl*2., Cvz=user->Fvz_cyl/user->A_cyl*2.;

    double Cpx=user->Fpx_cyl/user->A_cyl_x*4., Cpz=user->Fpz_cyl/user->A_cyl_z*4.;
    double Cvx=user->Fvx_cyl/user->A_cyl_x*4., Cvz=user->Fvz_cyl/user->A_cyl_z*4.;

    double Cdx=Cpx+Cvx, Cdz=Cpz+Cvz;

    if(!my_rank) {
      char filen[256];
      sprintf(filen, "%s/Force_Cylinder.dat", path);
      FILE *fp = fopen(filen, "a");
      if(ti==tistart) fprintf(fp, "\n\n***\n\n");
      fprintf(fp, "%d %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n", (int)ti, Cdx, 0., Cdz, Cpx, 0., Cpz, user->A_cyl_x, 0., user->A_cyl_z, user->A_cyl );
      fclose(fp);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
  }


  if(/*inletprofile==13 &&*/ ti) {

    if(!my_rank) {
      char filen[256];
      sprintf(filen, "%s/shear_velocity.dat", path);
      FILE *fp = fopen(filen, "a");
      fprintf(fp, "%d ", (int)ti);
      fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e\n",
              user->ustar_now[0],user->ustar_now[1],user->ustar_now[2],user->ustar_now[3],user->ustar_now[4],user->ustar_now[5]);
      fclose(fp);
    }
  }

  if(levelset) {
    DMDAVecRestoreArray(da, user->lDensity, &rho);
    DMDAVecRestoreArray(da, user->lLevelset, &level);
  }
  DMDAVecRestoreArray(da,user->lAj, &aj);
  DMDAVecRestoreArray(da, user->lP, &p);
  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(fda,user->lCsi, &csi);
  DMDAVecRestoreArray(fda,user->lEta, &eta);
  DMDAVecRestoreArray(fda,user->lZet, &zet);
  DMDAVecRestoreArray(fda,user->lUcat, &ucat);
  DMDAVecRestoreArray(fda,user->lUcont, &ucont);
  DMDAVecRestoreArray(fda,user->lCent, &cent);
}


void write_IBdata(UserCtx *user, IBMNodes *ibm)
{

  int ibi, i, j, k, elmt;

  int my_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);



  if (ti==tistart) {

    for(int m=0; m<nsave_IBpoints; m++) {
      double XX=save_IBcoor[m][0];
      double YY=save_IBcoor[m][1];
      double ZZ=save_IBcoor[m][2];

      double dis, dis_min;
      dis_min=1.e6;
      for(ibi=0;ibi<NumberOfBodies; ibi++)
        for(elmt=0;elmt<ibm[ibi].n_elmt;elmt++)
        {
          //dis=pow(XX-ibm[ibi].cent_x[elmt],2)+pow(YY-ibm[ibi].cent_y[elmt],2)+pow(ZZ-ibm[ibi].cent_z[elmt],2);
          dis=pow(XX-ibm[ibi].cent_x[elmt],2)+pow(YY-ibm[ibi].cent_y[elmt],2);

          if (dis<dis_min) {
            dis_min=dis;
            save_IBpoint[m][0]=ibi;
            save_IBpoint[m][1]=elmt;
          }

        }

    }


  }

  if (sediment && !my_rank) {
    for(int m=0; m<nsave_IBpoints; m++) {
      double XX=save_IBcoor[m][0];
      double YY=save_IBcoor[m][1];
      double ZZ=save_IBcoor[m][2];


      int _ibi=save_IBpoint[m][0];
      int _elmt=save_IBpoint[m][1];

      FILE *f;
      char filen[80];

      sprintf(filen, "%s/IBsavedData_%.2e_%.2e_%.2ed_dt_%g.dat", path, XX, YY, ZZ, user->dt);
      f = fopen(filen, "a");
      fprintf(f, "%d %.7e %.7e %.7e %.7e %.7e %.7e %.7e\n", (int)ti, ibm[_ibi].cent_x[_elmt], ibm[_ibi].cent_y[_elmt], ibm[_ibi].cent_z[_elmt], ibm[_ibi].Bvel[_elmt].x, ibm[_ibi].Bvel[_elmt].y, ibm[_ibi].Bvel[_elmt].z, ibm[_ibi].Shvel[_elmt]);
      if (LiveBed)fprintf(f, "%d %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e\n", (int)ti, ibm[_ibi].cent_x[_elmt], ibm[_ibi].cent_y[_elmt], ibm[_ibi].cent_z[_elmt], ibm[_ibi].Bvel[_elmt].x, ibm[_ibi].Bvel[_elmt].y, ibm[_ibi].Bvel[_elmt].z, ibm[_ibi].Shvel[_elmt], ibm[_ibi].C[_elmt]);
      fclose(f);

    }
  }
}



#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv)
{
/*
  #define F(x,y,z) (x*x*x +y*y+z)
  double val[3][3][3], w[3][3][3];
  double dx=1;
  for(int i=0; i<3; i++)
  for(int j=0; j<3; j++)
  for(int k=0; k<3; k++) {
  double x = (i+1)*dx;
  double y = (j+1)*dx;
  double z = (k+1)*dx;
  val[i][j][k] = F(x,y,z);
  w[i][j][k] = 10.0;
  }

  printf("%f %f\n",	integrate_testfilter(val, w), integrate_testfilter_simpson(val, w) ); //exact 16.33
  exit(0);

*/
  /*
    double L = 20., d=1., Beta=0.25, g=9.8, a=0.1; // L = width of the 3D tank
    double x = L/2., y=L/2.;
    double eta0 = a * exp ( -Beta * ( pow(x-L/2, 2) + pow(y-L/2, 2) ) );
    std::complex<double> I(0,1);
    std::complex<double> one(1,0);

    FILE *fp0 = fopen ("a.dat", "w");

    double dt=0.02;
    for(double _time=dt; _time<100.; _time+=dt) {
    std::complex<double> val(0,0);
    for(int m=0; m<40; m++)
    for(int n=0; n<40; n++) {
    double coeff_m=2., coeff_n=2.;
    double k_mn = sqrt( pow(M_PI/L,2.) * (m*m + n*n) );
    double omega_mn = sqrt ( g * k_mn * tanh (k_mn * d) );
    if(m==0) coeff_m=1.;
    if(n==0) coeff_n=1.;

    std::complex<double> Integral;
    Integral   = M_PI * a / (16*Beta) * exp ( -0.25*M_PI * ( 2.*(m+n)*I + (m*m+n*n)*M_PI/(Beta*L*L)*one ) );
    Integral *= ( one + exp(m*M_PI*I) ) * ( one + exp(n*M_PI*I) );
    Integral *= ERF ( (Beta*L*L - m*M_PI*I) * pow(2*sqrt(Beta)*L, -1) ) + ERF ( (Beta*L*L + m*M_PI*I) * pow(2*sqrt(Beta)*L, -1) );
    Integral *= ERF ( (Beta*L*L - n*M_PI*I) * pow(2*sqrt(Beta)*L, -1) ) + ERF ( (Beta*L*L + n*M_PI*I) * pow(2*sqrt(Beta)*L, -1) );

    double eta_mn = (1./(L*L)) * coeff_m * coeff_n * Integral.real();
    val += eta_mn * exp ( -I * omega_mn * _time ) * cos (n * M_PI / L * x) * cos (m * M_PI / L * y);
    }
    fprintf(fp0, "%e %e\n", _time, val.real());
    }
    fclose(fp0);
    exit(0);
  */
  /*
    double a = 0.01, b = 1, d = 1, g = 1.0; // 0.05
    double k2 = 2*M_PI/b, w2 = sqrt ( k2 * g * tanh (k2*d) ), k4 = 4*M_PI/b, w4 = sqrt ( k4 * g * tanh (k4*d) );
    const double x = 0.5;
    double eta0 = a * cos(k2*x), nu = 0.05;
    double kappa = g/nu/nu/k2/k2/k2;

    FILE *fp0 = fopen ("a.dat", "w");

    const std::complex<double> one(1.,0);
    double g1R=-1.106560764, g1I=1.768665503; // nu=0.02
    double g2R=1.106560764 , g2I=1.149250098;

    //double g1R = -5.590248751, g1I = 5.694713691; // for nu=0.001
    //double g2R = 5.590248751, g2I = 5.663214475;
    double delta = 4 * (6*g1R*g1R + 1) * (2*g1R*g1R + 1) - 4 * (kappa+1);
    double A1R = -2/delta * g1R, A2R = 2/delta * g1R;
    double A1I = -(2*pow(g1R,3.)+1.)/(g1R*g1I*delta), A2I = (2*pow(g1R,3.)-1.)/(g1R*g2I*delta);

    std::complex<double> g1(g1R, g1I), g2(g2R, g2I);
    std::complex<double> A1(A1R, A1I), A2(A2R, A2I);

    double dt=0.02;
    for(double _time=dt; _time<100.; _time+=dt) {
    std::complex<double> erf1 = ERF(g1*k2*sqrt(nu*_time)); // erf(g1*k2*sqrt(nu*t))
    std::complex<double> erf2 = ERF(g2*k2*sqrt(nu*_time));

    std::complex<double> val =  (A1/(one-g1*g1)) * (-g1*exp( (-one+g1*g1)*nu*k2*k2*_time ) * (one+erf1) + g1 + erf(k2*sqrt(nu*_time)) * one );
    val += (A2/(one-g2*g2)) * (-g2*exp( (-one+g2*g2)*nu*k2*k2*_time ) * (one+erf2) + g2 + erf(k2*sqrt(nu*_time)) * one );

    double exact = eta0 * ( 1 - 2.0 * kappa *  val.real() );
    fprintf(fp0, "%e %e\n", _time, exact);
    }
    fclose(fp0);
    exit(0);
  */
  /*
    double A=0.1, B=0;
    std::complex<double> a = ERF( std::complex<double>(A,B) );
    printf("%f, %f+%fi \n", erf(A),  a.real(), a.imag());

    exit(0);
  */
  /*
    extern double LSQ(double uwall, double sp, double *s, double *u, int n);

    double s[]={1, 2, 3};
    double u[]={6, 34, 102};


    printf("%f\n", LSQ(0, 2, s, u, 2));
    exit(0);
  */

  /*
    {
    double utau=0.01;

    int z;
    double y, ren=1.e4, dpdn=0;

    //utau = find_utau_Cabot(1./ren, 0.25, 0.01, 0.2, dpdn);
    //utau = find_utau_Werner(1./ren, 0.1, 0.001, 0.01);
    utau = 0.1;
    printf("u*=%f\n", utau);

    FILE *fp=fopen("./log.dat","w");
    y=1.e-4;
    for(z=1; z<100000; z++) {
    double u = u_Cabot(1./ren, y, utau, dpdn);
    double y_shift, yp_shift, ks;

    //ks = 1.;
    //yp_shift = 0.9*(sqrt(ks)-(ks)*exp(-ks/6.));
    //y_shift = (yp_shift)/utau/ren;
    double u10 = u_Cabot_roughness(1./ren, y, utau, dpdn, 0.01);//roughness ks_plus=10
    double u20 = u_Cabot_roughness(1./ren, y, utau, dpdn, 0.02);
    double u40 = u_Cabot_roughness(1./ren, y, utau, dpdn, 0.03);
    double u70 = u_Cabot_roughness(1./ren, y, utau, dpdn, 0.04);

    //double u = u_Werner(1./ren, y, utau);

    double yp = utau*y*ren;
    fprintf(fp, "%e %e %e %e %e %e %e\n", yp, u/utau, 2.5*log(yp)+5.5, u10/utau, u20/utau, u40/utau, u70/utau);

    //fprintf(fp, "%e %e\n", y, u);
    y+=1.e-4;
    if(y>1) break;
    }
    fclose(fp);

    Cmpnts Ua, Ub, Uc;
    double ustar, ks;
    Uc.x=0.1;
    Uc.y=0;
    Uc.z=0;
    Ua.x=0;
    Ua.y=0;
    Ua.z=0;

    ks=0;
    for(int l=0; l<20; l++) {
    wall_function_roughness(0.0001, ks, 0.01, 0.005, Ua, Uc, &Ub, &ustar, 0, 1, 0);
    printf("ks=%f, Ub.x=%f, Ub.y=%f, Ub.z=%f, u*=%f\n", ks, Ub.x, Ub.y, Ub.z, ustar);
    ks += 0.005;
    }
    exit(0);

    }
  */
  Vec	ResidualT;
  UserCtx	*user;


  PetscErrorCode	ierr;
  int	i, bi, ibi;

  PetscReal	norm;


  IBMNodes	*ibm, *ibm0, *ibm1;
  IBMInfo	*ibm_intp;

  // Added for fsi
  FSInfo        *fsi;
  PetscBool    DoSCLoop;
  int      itr_sc;

  int level;
  UserMG usermg;

  PetscBool	flg;
  //int tistart = 0;

  // begin add (xiaolei)
  ACL             *acl;   //xyang 3-18-2011
  IBMNodes        *wtm;   // xyang 04212011
  FSInfo        *fsi_wt;

  IBMNodes        *ibm_ACD;

  IBMNodes        *ibm_IBDelta;
  FSInfo        *fsi_IBDelta;
  // end add (xiaolei)

  //  int	ti;
  PetscInitialize(&argc, &argv, (char *)0, help);
  MPI_Barrier(PETSC_COMM_WORLD);

  MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);



  srand( time(NULL)) ;	// Seokkoo Kang

/*   PetscMalloc(sizeof(IBMNodes), &ibm0); */
/*   PetscMalloc(sizeof(IBMNodes), &ibm1); */

  //PetscOptionsInsertFile("control.dat");
  PetscOptionsInsertFile(PETSC_COMM_WORLD, "control.dat", PETSC_TRUE);



  PetscOptionsGetInt(PETSC_NULL, "-tio", &tiout, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-tiou", &tiout_ufield, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-tieu", &tiend_ufield, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-imm", &immersed, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-inv", &inviscid, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-rstart", &tistart, &rstart_flg);
  PetscOptionsGetInt(PETSC_NULL, "-imp", &implicit, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-imp_MAX_IT", &imp_MAX_IT, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-fsi", &movefsi, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-rfsi", &rotatefsi, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-radi", &radi, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-inlet", &inletprofile, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-str", &STRONG_COUPLING, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-rs_fsi", &rstart_fsi, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-cop", &cop, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-fish", &fish, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-mhv", &MHV, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-reg", &regime, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-twoD", &TwoD, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-thin", &thin, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-dgf_z", &dgf_z, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-dgf_y", &dgf_y, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-dgf_x", &dgf_x, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-dgf_az", &dgf_az, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-dgf_ay", &dgf_ay, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-dgf_ax", &dgf_ax, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-body", &NumberOfBodies, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-rbody", &NumberOfRotatingBodies, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-averaging", &averaging, PETSC_NULL);	// Seokkoo Kang: if 1 do averaging; always begin with -rstart 0
  PetscOptionsGetInt(PETSC_NULL, "-phase_averaging", &phase_averaging, PETSC_NULL);	// Seokkoo Kang: period of phase averaging
  phase_averaging = std::max(phase_averaging, (PetscInt)0);
  if(averaging==0) phase_averaging=0;

  PetscOptionsGetInt(PETSC_NULL, "-binary", &binary_input, PETSC_NULL);	// Seokkoo Kang: if 1 binary PLOT3D file, if 0 ascii.
  PetscOptionsGetInt(PETSC_NULL, "-xyz", &xyz_input, PETSC_NULL);			// Seokkoo Kang: if 1 text xyz format, useful for very big (>1GB) Cartesian grid
  PetscOptionsGetInt(PETSC_NULL, "-les", &les, PETSC_NULL);				// Seokkoo Kang: if 1 Smagorinsky with Cs=0.1, if 2 Dynamic model
  PetscOptionsGetInt(PETSC_NULL, "-inlet_buffer_k", &inlet_buffer_k, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-wallfunction", &wallfunction, PETSC_NULL);	// Seokkoo Kang: 1 or 2
  PetscOptionsGetInt(PETSC_NULL, "-slipbody", &slipbody, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-central", &central, PETSC_NULL);//central differencing
  PetscOptionsGetInt(PETSC_NULL, "-second_order", &second_order, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-weno", &weno, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-initialzero", &initialzero, PETSC_NULL);	// Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-freesurface", &freesurface, PETSC_NULL);	// Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-rans", &rans, PETSC_NULL);			// Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-cross_diffusion", &cross_diffusion, PETSC_NULL);                     // Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-lowRe", &lowRe, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-stension", &surface_tension, PETSC_NULL);			// Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-subdt_levelset", &subdt_levelset, PETSC_NULL);			// Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-delete", &delete_previous_file, PETSC_NULL);	// Seokkoo Kang: delete previous time step's saved file for saving disk space
  PetscOptionsGetInt(PETSC_NULL, "-mixed", &mixed, PETSC_NULL);			// Seokkoo Kang: mixed model option for LES
  PetscOptionsGetInt(PETSC_NULL, "-clark", &clark, PETSC_NULL);			// Seokkoo Kang: mixed model option for LES
  PetscOptionsGetInt(PETSC_NULL, "-vorticity", &vorticity, PETSC_NULL);			// Seokkoo Kang: vorticity form for viscous terms
  PetscOptionsGetInt(PETSC_NULL, "-pseudo", &pseudo_periodic, PETSC_NULL);	// Seokkoo Kang: pseudo periodic BC in k-direction for genenration of inflow condition

  PetscOptionsGetInt(PETSC_NULL, "-conv_diff", &conv_diff, PETSC_NULL);		// Ali
  PetscOptionsGetInt(PETSC_NULL, "-density_current", &density_current, PETSC_NULL);		// Ali
  PetscOptionsGetInt(PETSC_NULL, "-SuspendedParticles", &SuspendedParticles, PETSC_NULL);	// Ali
  PetscOptionsGetReal(PETSC_NULL, "-w_s", &w_s, PETSC_NULL);			// Ali
  PetscOptionsGetReal(PETSC_NULL, "-background_conc", &Background_Conc, PETSC_NULL);			// Ali
  PetscOptionsGetReal(PETSC_NULL, "-inlet_conc", &Inlet_Conc, PETSC_NULL);			// Ali

  PetscOptionsGetInt(PETSC_NULL, "-levelset", &levelset, PETSC_NULL);     // Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-fix_level", &fix_level, PETSC_NULL);     // Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-levelset_it", &levelset_it, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-sloshing", &sloshing, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-bubble", &bubble, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-rotdir", &rotdir, PETSC_NULL);


  PetscOptionsGetInt(PETSC_NULL, "-i_periodic", &i_periodic, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-j_periodic", &j_periodic, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-k_periodic", &k_periodic, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-laplacian", &laplacian, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-qcrout", &qcr, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-ii_periodic", &ii_periodic, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-jj_periodic", &jj_periodic, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-kk_periodic", &kk_periodic, PETSC_NULL);

  periodic = i_periodic+j_periodic+k_periodic+ii_periodic+jj_periodic+kk_periodic;

  PetscOptionsGetInt(PETSC_NULL, "-perturb", &initial_perturbation, PETSC_NULL);	// Seokkoo Kang: give a random perturbation for initial condition
  PetscOptionsGetInt(PETSC_NULL, "-skew", &skew, PETSC_NULL);				// Seokkoo Kang: skew symmetric form of advection term
  PetscOptionsGetInt(PETSC_NULL, "-dynamic_freq", &dynamic_freq, PETSC_NULL);		// Seokkoo Kang: LES dynamic compute frequency
  if(dynamic_freq<1) dynamic_freq=1;

  PetscOptionsGetInt(PETSC_NULL, "-save_inflow", &save_inflow, PETSC_NULL);		// Seokkoo Kang: save infow BC to files; should be used in conjunction with -pseudo 1
  PetscOptionsGetInt(PETSC_NULL, "-save_inflow_period", &save_inflow_period, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-save_inflow_minus", &save_inflow_minus, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-ti_lastsave", &ti_lastsave, PETSC_NULL);

  read_inflow_period = save_inflow_period;
  PetscOptionsGetInt(PETSC_NULL, "-read_inflow_period", &read_inflow_period, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-localstep", &localstep, PETSC_NULL);		// Seokkoo Kang: localstep ( explict + implicit momentum solver )
  PetscOptionsGetInt(PETSC_NULL, "-recycle", &inflow_recycle_perioid, PETSC_NULL);	// Seokkoo Kang, set recycling period of the inflow data
  PetscOptionsGetInt(PETSC_NULL, "-save_memory", &save_memory, PETSC_NULL);	// Seokkoo Kang, save_memory
  PetscOptionsGetInt(PETSC_NULL, "-ibm_search", &ibm_search, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-ip", &i_proc, PETSC_NULL);			// Seokkoo Kang: number of processors in i direction
  PetscOptionsGetInt(PETSC_NULL, "-jp", &j_proc, PETSC_NULL);			// Seokkoo Kang: number of processors in j direction
  PetscOptionsGetInt(PETSC_NULL, "-kp", &k_proc, PETSC_NULL);			// Seokkoo Kang: number of processors in k direction
  PetscOptionsGetInt(PETSC_NULL, "-poisson", &poisson, PETSC_NULL); 	// Seokkoo Kang
  PetscOptionsGetInt(PETSC_NULL, "-amg_agg", &amg_agg, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-amg_coarsentype", &amg_coarsentype, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-testfilter_ik", &testfilter_ik, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-testfilter_1d", &testfilter_1d, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-poisson_it", &poisson_it, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-i_homo_filter", &i_homo_filter, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-j_homo_filter", &j_homo_filter, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-k_homo_filter", &k_homo_filter, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-fix_outlet", &fix_outlet, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-fix_inlet", &fix_inlet, PETSC_NULL);


  // add begin (xiaolei)

  PetscOptionsGetInt(PETSC_NULL, "-rotor_modeled", &rotor_model, PETSC_NULL); // xyang 12-7-2010
  PetscOptionsGetReal(PETSC_NULL, "-indf_ax", &indf_ax, PETSC_NULL); // xyang 12-16-2010

  PetscOptionsGetInt(PETSC_NULL, "-imin_wm", &imin_wm, PETSC_NULL); // xyang 1-11-2011
  PetscOptionsGetInt(PETSC_NULL, "-imax_wm", &imax_wm, PETSC_NULL); // xyang 1-11-2011

  PetscOptionsGetInt(PETSC_NULL, "-jmin_wm", &jmin_wm, PETSC_NULL); // xyang 1-11-2011
  PetscOptionsGetInt(PETSC_NULL, "-jmax_wm", &jmax_wm, PETSC_NULL); // xyang 1-11-2011

  PetscOptionsGetInt(PETSC_NULL, "-kmin_wm", &kmin_wm, PETSC_NULL); // xyang 1-11-2011
  PetscOptionsGetInt(PETSC_NULL, "-kmax_wm", &kmax_wm, PETSC_NULL); // xyang 1-11-2011

  PetscOptionsGetInt(PETSC_NULL, "-imin_wmtmprt", &imin_wmtmprt, PETSC_NULL); // xyang 10-22-2012
  PetscOptionsGetInt(PETSC_NULL, "-imax_wmtmprt", &imax_wmtmprt, PETSC_NULL); // xyang 10-22-2012

  PetscOptionsGetInt(PETSC_NULL, "-jmin_wmtmprt", &jmin_wmtmprt, PETSC_NULL); // xyang 10-22-2012
  PetscOptionsGetInt(PETSC_NULL, "-jmax_wmtmprt", &jmax_wmtmprt, PETSC_NULL); // xyang 10-22-2012

  PetscOptionsGetInt(PETSC_NULL, "-kmin_wmtmprt", &kmin_wmtmprt, PETSC_NULL); // xyang 10-22-2012
  PetscOptionsGetInt(PETSC_NULL, "-kmax_wmtmprt", &kmax_wmtmprt, PETSC_NULL); // xyang 10-22-2012


  PetscOptionsGetInt(PETSC_NULL, "-surface_p_out", &surface_p_out, PETSC_NULL); // xyang 3-2-2011

  PetscOptionsGetInt(PETSC_NULL, "-num_blade", &num_blade, PETSC_NULL); // xyang 3-2-2011

  PetscOptionsGetInt(PETSC_NULL, "-num_foiltype", &num_foiltype, PETSC_NULL); // xyang 3-18-2011

//        PetscOptionsGetInt(PETSC_NULL, "-num_innergrid", &num_innergrid, PETSC_NULL); // xyang 3-18-2011

  PetscOptionsGetReal(PETSC_NULL, "-dh1_wm", &dh1_wm, PETSC_NULL); //xyang 4-11-2011
  PetscOptionsGetReal(PETSC_NULL, "-dhratio_wm", &dhratio_wm, PETSC_NULL); //xyang 4-11-2011

  PetscOptionsGetInt(PETSC_NULL, "-rotatewt", &rotatewt, PETSC_NULL); // xyang 3-18-2011

  PetscOptionsGetInt(PETSC_NULL, "-IB_delta", &IB_delta, PETSC_NULL); // xyang 3-18-2011
  PetscOptionsGetInt(PETSC_NULL, "-NumIBPerLoc", &NumIBPerLoc, PETSC_NULL); // xyang 3-18-2011
  PetscOptionsGetReal(PETSC_NULL, "-reflength_wt", &reflength_wt, PETSC_NULL); // xyang 12-16-2010
  PetscOptionsGetReal(PETSC_NULL, "-reflength_IBDelta", &reflength_IBDelta, PETSC_NULL); // xyang 12-16-2010
  PetscOptionsGetReal(PETSC_NULL, "-tipspeedratio", &tipspeedratio, PETSC_NULL); // xyang 2-12-2012
  PetscOptionsGetReal(PETSC_NULL, "-r_rotor", &r_rotor, PETSC_NULL); // xyang 2-12-2012
  PetscOptionsGetReal(PETSC_NULL, "-r_nacelle", &r_nacelle, PETSC_NULL); // xyang 2-12-2012

  PetscOptionsGetInt(PETSC_NULL, "-IB_wm", &IB_wm, PETSC_NULL); // xyang 6-3-2011
  PetscOptionsGetInt(PETSC_NULL, "-IB_wmtmprt", &IB_wmtmprt, PETSC_NULL); // xyang 10-22-2012

  PetscOptionsGetInt(PETSC_NULL, "-temperature", &temperature, PETSC_NULL); // xyang 6-3-2011

  PetscOptionsGetInt(PETSC_NULL, "-deltafunc", &deltafunc, PETSC_NULL); // xyang 6-3-2011

  PetscOptionsGetReal(PETSC_NULL, "-prt_eps", &prt_eps, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-i_periodicIB", &i_periodicIB, PETSC_NULL); // xyang 6-3-2011
  PetscOptionsGetInt(PETSC_NULL, "-j_periodicIB", &j_periodicIB, PETSC_NULL); // xyang 6-3-2011
  PetscOptionsGetInt(PETSC_NULL, "-k_periodicIB", &k_periodicIB, PETSC_NULL); // xyang 6-3-2011

  PetscOptionsGetInt(PETSC_NULL, "-Force_wm", &Force_wm, PETSC_NULL); // xyang 10-24-2012

  PetscOptionsGetInt(PETSC_NULL, "-Shear_wm", &Shear_wm, PETSC_NULL); // xyang 10-24-2012

  PetscOptionsGetInt(PETSC_NULL, "-add_fluc", &add_fluctuations, PETSC_NULL); // xyang 11-03-2011
  PetscOptionsGetInt(PETSC_NULL, "-add_fluc_tmprt", &add_fluctuations_tmprt, PETSC_NULL); // xyang 11-03-2011
  PetscOptionsGetInt(PETSC_NULL, "-les_prt", &les_prt, PETSC_NULL); // xyang 11-03-2011

  PetscOptionsGetInt(PETSC_NULL, "-infRe", &infRe, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-MoveFrame", &MoveFrame, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-u_frame", &u_frame, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-v_frame", &v_frame, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-w_frame", &w_frame, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-C_iww", &C_iww, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-a_iww", &a_iww, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-lamda_iww", &lamda_iww, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-xmin", &xmin, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-xmax", &xmax, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-ymin", &ymin, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-ymax", &ymax, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-zmin", &zmin, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-zmax", &zmax, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-ii_periodicWT", &ii_periodicWT, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-jj_periodicWT", &jj_periodicWT, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-kk_periodicWT", &kk_periodicWT, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-Nx_WT", &Nx_WT, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-Ny_WT", &Ny_WT, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-Nz_WT", &Nz_WT, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-Sx_WT", &Sx_WT, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-Sy_WT", &Sy_WT, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-Sz_WT", &Sz_WT, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-New_wallmodel", &New_wallmodel, PETSC_NULL);


  PetscOptionsGetInt(PETSC_NULL, "-turbine", &NumberOfTurbines, PETSC_NULL);  // xyang
  PetscOptionsGetInt(PETSC_NULL, "-NumberOfIBDelta", &NumberOfIBDelta, PETSC_NULL);  // xyang


  PetscOptionsGetInt(PETSC_NULL, "-SpongeLayer", &SpongeLayer, PETSC_NULL);  // xyang
  PetscOptionsGetInt(PETSC_NULL, "-SpongeDistance", &SpongeDistance, PETSC_NULL);  // xyang

  PetscOptionsGetInt(PETSC_NULL, "-MoveCylinderTest", &MoveCylinderTest, PETSC_NULL);  // xyang

  PetscOptionsGetInt(PETSC_NULL, "-inflow_levelset", &inflow_levelset, PETSC_NULL);  // xyang

  PetscOptionsGetInt(PETSC_NULL, "-sublevel", &sublevel, PETSC_NULL);  // xyang
  PetscOptionsGetReal(PETSC_NULL, "-smoothlevel", &smoothlevel, PETSC_NULL);  // xyang

  PetscOptionsGetReal(PETSC_NULL, "-dfunc_wd", &dfunc_wd, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-FixTipSpeedRatio", &FixTipSpeedRatio, PETSC_NULL);  // xyang

  PetscOptionsGetReal(PETSC_NULL, "-dt_bed", &dt_bed, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-particle_dens", &particle_dens, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-particle_D50", &particle_D50, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-smooth_bed", &smooth_bed, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-bed_avg", &bed_avg, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-particlevel_model", &particlevel_model, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-LS_bedConstr", &LS_bedConstr, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-Angle_repose", &Angle_repose, PETSC_NULL);
  Angle_repose	  *= M_PI;
  Angle_repose	  /= 180.;
  PetscOptionsGetReal(PETSC_NULL, "-bed_porosity", &bed_porosity, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-sb_bed", &sb_bed, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, "-number_smooth", &number_smooth, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-deltab", &deltab, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-rs_bed", &rs_bed, PETSC_NULL);
  // add end (xiaolei)

  // sediment begin
  PetscOptionsGetInt(PETSC_NULL, "-sediment", &sediment, PETSC_NULL); //ali
  PetscOptionsGetInt(PETSC_NULL, "-periodic_morpho", &periodic_morpho, PETSC_NULL); //ali
  PetscOptionsGetReal(PETSC_NULL, "-Nseg", &Nseg, PETSC_NULL); //ali
  PetscOptionsGetInt(PETSC_NULL, "-livebed", &LiveBed, PETSC_NULL); //ali
  PetscOptionsGetInt(PETSC_NULL, "-rigidbed", &RigidBed, PETSC_NULL); //ali
  PetscOptionsGetInt(PETSC_NULL, "-zero_grad", &zero_grad, PETSC_NULL); //ali
  PetscOptionsGetInt(PETSC_NULL, "-projection_method", &projection_method, PETSC_NULL); //ali
  PetscOptionsGetInt(PETSC_NULL, "-bed_roughness", &bed_roughness, PETSC_NULL);	// ali: 0 or 1
  PetscOptionsGetReal(PETSC_NULL, "-k_ss", &k_ss, PETSC_NULL);	// ali
  PetscOptionsGetInt(PETSC_NULL, "-ib_depth", &input_ib_depth, PETSC_NULL); //ali

  // sediment end

  if(movefsi || rotatefsi) save_memory=0;
  if(!rotatefsi && !movefsi) rstart_fsi=0; //seokkoo

  sprintf(path, ".");
  PetscOptionsGetString(PETSC_NULL,"-path", path, 256, PETSC_NULL);		//  Seokkoo Kang: path for saving output; grid.dat, bcs,dat should be put there, but control.dat should exist in the current directory where job is submitted

  sprintf(gridfile, "grid.dat");
  PetscOptionsGetString(PETSC_NULL,"-grid", gridfile, 256, PETSC_NULL);	//  Seokkoo Kang: the name of the grid file other than grid.dat if you want


  sprintf(path_inflow, "./inflow");
  PetscOptionsGetString(PETSC_NULL,"-path_inflow", path_inflow, 256, PETSC_NULL);         //  Xiaolei Yang: path for inflow field

  /*
  // test !!!!!!!!!!!!!!!!!!!!!
  double a[6][6], b[6], c[6], d[6];
  b[0]=1;
  b[1]=1;
  b[2]=1;
  b[3]=1;
  b[4]=1;
  b[5]=1;
  a[0][0]=1; a[0][1]=0.5; a[0][2]=0; a[0][3]=0.6; a[0][4]=0; a[0][5]=0;
  a[1][0]=0.5; a[1][1]=1; a[1][2]=0.5; a[1][3]=0; a[1][4]=0; a[1][5]=0;
  a[2][0]=0.4; a[2][1]=0.5; a[2][2]=1; a[2][3]=0.5; a[2][4]=0; a[2][5]=0;
  a[3][0]=0; a[3][1]=0.6; a[3][2]=0.5; a[3][3]=1; a[3][4]=0.5; a[3][5]=0;
  a[4][0]=0; a[4][1]=0; a[4][2]=10; a[4][3]=0.5; a[4][4]=1; a[4][5]=0.5;
  a[5][0]=0; a[5][1]=0.001; a[5][2]=0; a[5][3]=90; a[5][4]=0.5; a[5][5]=1;


  PetscPrintf(PETSC_COMM_WORLD, "b0=%le b1=%le b2=%le b3=%le b4=%le b5=%le !!!!!! \n", b[0], b[1], b[2], b[3], b[4], b[5]);

  AGAUS (a, b, c);
  PetscPrintf(PETSC_COMM_WORLD, "c0=%le c1=%le c2=%le c3=%le c4=%le c5=%le !!!!!! \n", c[0], c[1], c[2], c[3], c[4], c[5]);


  a[0][0]=1; a[0][1]=0.5; a[0][2]=0; a[0][3]=0.6; a[0][4]=0; a[0][5]=0;
  a[1][0]=0.5; a[1][1]=1; a[1][2]=0.5; a[1][3]=0; a[1][4]=0; a[1][5]=0;
  a[2][0]=0.4; a[2][1]=0.5; a[2][2]=1; a[2][3]=0.5; a[2][4]=0; a[2][5]=0;
  a[3][0]=0; a[3][1]=0.6; a[3][2]=0.5; a[3][3]=1; a[3][4]=0.5; a[3][5]=0;
  a[4][0]=0; a[4][1]=0; a[4][2]=10; a[4][3]=0.5; a[4][4]=1; a[4][5]=0.5;
  a[5][0]=0; a[5][1]=0.001; a[5][2]=0; a[5][3]=90; a[5][4]=0.5; a[5][5]=1;



  int ii,jj;

  for(jj=0;jj<6;jj++) {
  d[jj]=0.0;

  for(ii=0;ii<6;ii++) {
  d[jj]+=a[jj][ii]*c[ii];
  }

  }



  PetscPrintf(PETSC_COMM_WORLD, "d0=%le d1=%le d2=%le d3=%le d4=%le d5=%le !!!!!! \n", d[0], d[1], d[2], d[3], d[4], d[5]);

  exit(0);
  // test !!!!!!!!!!!!!!!!!!!!!!!
  */

  int len=strlen(path);
  if(path[len-1]=='/') path[len-1]=0;

  {
    char saveoption_file[400];
    sprintf(saveoption_file, "%s/savepoints", path);
    FILE *fp=fopen(saveoption_file, "r");

    if(fp!=NULL) {
      int i=0;
      do {
        fscanf(fp, "%le %le %le\n", &save_coor[i][0], &save_coor[i][1], &save_coor[i][2]); // xiaolei
        i++;
      } while(!feof(fp));
      nsave_points=i;
      fclose(fp);
    }
  }


  // xiaolei add
  if(immersed)
  {
    char saveoption_file[400];
    sprintf(saveoption_file, "%s/saveIBpoints", path);
    FILE *fp=fopen(saveoption_file, "r");

    if(fp!=NULL) {
      int i=0;
      do {
        fscanf(fp, "%le %le %le\n", &save_IBcoor[i][0], &save_IBcoor[i][1], &save_IBcoor[i][2]);
        i++;
      } while(!feof(fp));
      nsave_IBpoints=i;
      fclose(fp);
    }
  }

  // end add



  if(save_inflow) {
    char save_ksection_file[400];
    sprintf(save_ksection_file, "%s/savekplanes", path);
    FILE *fp=fopen(save_ksection_file, "r");

    if(fp!=NULL) {
      int i=0;
      do {
        fscanf(fp, "%d\n", &save_ksection[i]);
        i++;
      } while(!feof(fp));
      nsave_ksection=i;
      fclose(fp);
    }
  }


  if(TwoD) PetscPrintf(PETSC_COMM_WORLD, "\n\n!!! 2D computation !!! \n\n");
  if(i_periodic) PetscPrintf(PETSC_COMM_WORLD, "\nI-Periodic\n");
  if(ii_periodic) PetscPrintf(PETSC_COMM_WORLD, "\nII-Periodic\n");
  if(j_periodic) PetscPrintf(PETSC_COMM_WORLD, "\nJ-Periodic \n");
  if(jj_periodic) PetscPrintf(PETSC_COMM_WORLD, "\nJJ-Periodic \n");
  if(k_periodic) PetscPrintf(PETSC_COMM_WORLD, "\nK-Periodic \n");
  if(kk_periodic) PetscPrintf(PETSC_COMM_WORLD, "\nKK-Periodic \n");


  //PetscOptionsGetReal(PETSC_NULL, "-Fr", &Fr, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-fluct_rms", &fluct_rms, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-max_cs", &max_cs, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-flux", &inlet_flux, PETSC_NULL);			// Seokkoo Kang: the amount of inlet flux, if not set mean bulk velocity is set to 1
  PetscOptionsGetReal(PETSC_NULL, "-imp_tol", &imp_free_tol, PETSC_NULL);		// Seokkoo Kang: tolerance of implicit matrix free solver. 1.e-4 is enough for most cases.
  PetscOptionsGetReal(PETSC_NULL, "-poisson_tol", &poisson_tol, PETSC_NULL);		// Seokkoo Kang: tolerance of implicit matrix free solver. 1.e-4 is enough for most cases.
  PetscOptionsGetReal(PETSC_NULL, "-les_eps", &les_eps, PETSC_NULL);		// Seokkoo Kang: small value for preventing very large Cs values in les>1
  PetscOptionsGetReal(PETSC_NULL, "-dpdz", &mean_pressure_gradient, &dpdz_set);		// Seokkoo Kang: tolerance of implicit matrix free solver. 1.e-4 is enough for most cases.
//	if(fabs(dpdz_set)<1.e-10) dpdz_set=PETSC_FALSE; 				// deactivate xiaolei
  PetscOptionsGetReal(PETSC_NULL, "-roughness", &roughness_size, &rough_set);	// Seokkoo Kang: roughness_size
  PetscOptionsGetReal(PETSC_NULL, "-amg_thresh", &amg_thresh, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-rho0", &rho_water, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-rho1", &rho_air, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-mu0", &mu_water, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-mu1", &mu_air, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-dthick", &dthick, &dthick_set);
  PetscOptionsGetReal(PETSC_NULL, "-dthick_const", &dthick_const, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-dtau_ratio", &dtau_ratio, PETSC_NULL); //for levelset reinit
  PetscOptionsGetReal(PETSC_NULL, "-d_parameter", &d_parameter, PETSC_NULL); //for levelset 2

  PetscPrintf(PETSC_COMM_WORLD, "\nrho0=%f, rho1=%f, mu0=%f, mu1=%f\n", rho_water, rho_air, mu_water, mu_air);

  PetscOptionsGetReal(PETSC_NULL, "-gx", &gravity_x, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-gy", &gravity_y, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-gz", &gravity_z, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-x_r", &(x_r), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-y_r", &(y_r), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-z_r", &(z_r), PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-inlet_y", &inlet_y, &inlet_y_flag);
  PetscOptionsGetReal(PETSC_NULL, "-outlet_y", &outlet_y, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-inlet_z", &inlet_z, &inlet_z_flag);
  PetscOptionsGetReal(PETSC_NULL, "-outlet_z", &outlet_z, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-x_c", &(CMx_c), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-y_c", &(CMy_c), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-z_c", &(CMz_c), PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-imp_atol", &(imp_atol), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-imp_rtol", &(imp_rtol), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-imp_stol", &(imp_stol), PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-angvel", &angvel, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD, "angvel=%f\n", angvel);

  PetscOptionsGetReal(PETSC_NULL, "-sloshing_a", &sloshing_a, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-sloshing_b", &sloshing_b, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-sloshing_d", &sloshing_d, PETSC_NULL);

  PetscOptionsGetReal(PETSC_NULL, "-bubble_d", &bubble_d, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-bubble_z", &bubble_z, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, "-bubble_ws", &bubble_ws, PETSC_NULL);

  if (fish) {
    PetscOptionsGetReal(PETSC_NULL, "-St_exp", &(St_exp), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-wlngth", &(wavelength), PETSC_NULL);
  }

  PetscPrintf(PETSC_COMM_WORLD, "tiout %d %le %le thin %d!\n",tiout, imp_atol,imp_rtol,thin);

  if (MHV)
    L_dim=1./25.4;//.005;
  else
    L_dim=1.;

  if (MHV) NumberOfBodies=3;//2;

  if (immersed) {
    PetscMalloc(NumberOfBodies*sizeof(IBMNodes), &ibm);
    PetscMalloc(NumberOfBodies*sizeof(FSInfo), &fsi);
    ibm_ptr = ibm;
    fsi_ptr = fsi;
  }


  // add begin (xiaolei)
  if (IB_delta) {
    PetscMalloc(NumberOfIBDelta*sizeof(IBMNodes), &ibm_IBDelta);
    PetscMalloc(NumberOfIBDelta*sizeof(FSInfo), &fsi_IBDelta);
  }


  if (rotor_model) {
    PetscMalloc(NumberOfTurbines*sizeof(IBMNodes), &wtm);
    if ((rotor_model == 3 || rotor_model == 2) && FixTipSpeedRatio)  PetscMalloc(NumberOfTurbines*sizeof(IBMNodes), &ibm_ACD);
    PetscMalloc(NumberOfTurbines*sizeof(FSInfo), &fsi_wt);

    for (i=0;i<NumberOfTurbines;i++) {
      fsi_wt[i].angvel_x=0;
      fsi_wt[i].angvel_y=0;
      fsi_wt[i].angvel_z=0;
      fsi_wt[i].angvel_axis=angvel;
    }
  }

  if (rotor_model == 3 || rotor_model == 2) {
    PetscMalloc(num_foiltype*sizeof(ACL), &acl);
  }

  // add end (xiaolei)


  MG_Initial(&usermg, ibm);


  // Seokkoo Kang
  level = usermg.mglevels-1;
  user = usermg.mgctx[level].user;

  VecDuplicate(user->lP, &user->lUstar);

  VecDuplicate(user->lP, &user->lTau); // add (xiaolei)
  if (temperature) VecDuplicate(user->lP, &user->lQ_scalar); // add (xiaolei)


  if(averaging) {	// Seokkoo Kang
    level = usermg.mglevels-1;
    user = usermg.mgctx[level].user;

    VecDuplicate(user->Ucat, &user->Ucat_sum);
    VecDuplicate(user->Ucat, &user->Ucat_cross_sum);
    VecDuplicate(user->Ucat, &user->Ucat_square_sum);
    VecDuplicate(user->P, &user->P_sum);
    /*if(averaging>=2)*/ VecDuplicate(user->P, &user->P_square_sum);

    VecSet(user->Ucat_sum,0);
    VecSet(user->Ucat_cross_sum,0);
    VecSet(user->Ucat_square_sum,0);
    VecSet(user->P_sum,0);
    /*if(averaging>=2) */ VecSet(user->P_square_sum,0);


    // add begin (xiaolei)
    if(temperature) {
      VecDuplicate(user->P, &user->T_sum);
      VecSet(user->T_sum, 0.);
      VecDuplicate(user->P, &user->TP_sum);
      VecSet(user->TP_sum, 0.);
      if (les_prt) {
        VecDuplicate(user->P, &user->Prt_sum);
        VecSet(user->Prt_sum, 0.);
      }
      VecDuplicate(user->P, &user->TT_sum);
      VecSet(user->TT_sum, 0.);
      VecDuplicate(user->Ucont, &user->TU_sum);
      VecSet(user->TU_sum, 0.);
    }
    // add end (xiaolei)

    if(phase_averaging) {
      VecDuplicate(user->Ucat, &user->Ucat_sum_phase);
      VecDuplicate(user->Ucat, &user->Ucat_cross_sum_phase);
      VecDuplicate(user->Ucat, &user->Ucat_square_sum_phase);
      VecDuplicate(user->P, &user->P_sum_phase);
      VecDuplicate(user->P, &user->P_square_sum_phase);

      VecSet(user->Ucat_sum_phase,0);
      VecSet(user->Ucat_cross_sum_phase,0);
      VecSet(user->Ucat_square_sum_phase,0);
      VecSet(user->P_sum_phase,0);
      VecSet(user->P_square_sum_phase,0);
    }

    if(levelset) {
      VecDuplicate(user->P, &user->Levelset_sum);
      VecDuplicate(user->P, &user->Levelset_square_sum);
      VecSet(user->Levelset_sum, 0.);
      VecSet(user->Levelset_square_sum, 0.);
    }

    if(rans) {
      VecDuplicate(user->P, &user->K_sum);
      VecSet(user->K_sum, 0.);
    }

    if (conv_diff)
    {
      VecDuplicate(user->P, &user->Conc_sum);
      VecSet(user->Conc_sum, 0.);
      VecDuplicate(user->Ucat, &user->Conc_cross_sum);
      VecSet(user->Conc_cross_sum, 0.);
    }

    /*
      if(les) {
      VecDuplicate(user->P, &user->Nut_sum);
      VecSet(user->Nut_sum, 0.);
      }*/

    if(averaging>=3) {
      /*
        if(les) {
        VecDuplicate(user->P, &user->tauS_sum);
        VecSet(user->tauS_sum, 0);
        }
      */
      VecDuplicate(user->P, &user->Udp_sum);
      VecDuplicate(user->P, &user->dU2_sum); // was Ucont before
      VecDuplicate(user->Ucont, &user->UUU_sum);
      VecDuplicate(user->Ucont, &user->Vort_sum);
      VecDuplicate(user->Ucont, &user->Vort_square_sum);

      VecSet(user->Udp_sum, 0);
      VecSet(user->dU2_sum, 0);
      VecSet(user->UUU_sum, 0);
      VecSet (user->Vort_sum, 0);
      VecSet (user->Vort_square_sum, 0);
    }
  }

  // Seokkoo Kang
#ifdef DIRICHLET
  if(freesurface) {
    extern void Initialize_free_surface(UserCtx *user);
    level = usermg.mglevels-1;
    user = usermg.mgctx[level].user;
    for (bi=0; bi<block_number; bi++)  Initialize_free_surface(&user[bi]);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD, "\n**************** Warning !! Freesurface option not set *********************************!\n");
  }
#endif

  if (immersed) {
    level = usermg.mglevels-1;
    user = usermg.mgctx[level].user;
    for (bi=0; bi<block_number; bi++) {
      PetscMalloc(NumberOfBodies*sizeof(IBMList), &(user[bi].ibmlist));
      for (ibi=0;ibi<NumberOfBodies;ibi++) {
	InitIBMList(&(user[bi].ibmlist[ibi]));
      }
    }

    if (MHV) {
      i=0;
      // read casing
      CMz_c=0.;//1.105;
      ibm_read_ucd(&ibm[i], i);
      // read valves
      CMz_c=4.49+0.31;
      CMy_c=.0;
      L_dim=1./28.;


      for (ibi=1; ibi<NumberOfBodies; ibi++) {
	if (ibi==2) CMy_c=-CMy_c;
	ibm_read_ucd(&ibm[ibi], ibi);
	PetscPrintf(PETSC_COMM_WORLD, "Ibm read MHV!\n");

	FsiInitialize(0, &fsi[ibi], ibi);
      }

      fsi[1].y_c = -0.0878; fsi[1].z_c = 4.143;//4.21;
      fsi[2].y_c =  0.0878; fsi[2].z_c = 4.143;//4.21;

      fsi[1].S_ang_n[0]= max_angle; fsi[1].S_ang_r[0]= max_angle; fsi[1].S_ang_r[0]= max_angle;  fsi[1].S_ang_rm1[0]= max_angle;
      fsi[2].S_ang_n[0]= -max_angle; fsi[2].S_ang_r[0]= -max_angle; fsi[2].S_ang_r[0]= -max_angle;  fsi[2].S_ang_rm1[0]= -max_angle;

      for (ibi=1; ibi<NumberOfBodies; ibi++) {
	Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi],0.,ibi);
      }
      MPI_Barrier(PETSC_COMM_WORLD);
      for (ibi=0; ibi<NumberOfBodies; ibi++) {
	ibm_surface_out(&ibm[ibi], 0, ibi);
      }
    } else {
      for (i=0;i<NumberOfBodies;i++) {

	PetscPrintf(PETSC_COMM_WORLD, "Ibm Reading %d!\n", i);
	/*     ibm_read(ibm0); */
	if (sediment) ibm_read_ucd_sedi(&ibm[i], i);
	else ibm_read_ucd(&ibm[i], i);
	MPI_Barrier(PETSC_COMM_WORLD);
	PetscPrintf(PETSC_COMM_WORLD, "Ibm Read %d!\n", i);
	// init for fsi
/* 	FsiInitialize(ibm[i].n_elmt, &fsi[i], i); */
	FsiInitialize(0, &fsi[i], i);
      }
    }
  }

  if (immersed) {
    ti = 0;
    if (rstart_flg) ti = tistart;
    //fluxin(&(usermg.mgctx[0].user[0]));
/*     Elmt_Move(ibm0, &(usermg.mgctx[0].user[0])); */
/*     Init_Elmt(&ibm, ibm0, ibm1); */
  }



  if (rotor_model) {

    PetscReal cl = 1.;

    PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &cl, PETSC_NULL);


    if (!my_rank) {

      FILE *fd;
      char str[256];
      sprintf(str, "%s/CenterWT.dat", path);
      fd = fopen(str, "r");
      if(!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(1);

      for (ibi=0;ibi<NumberOfTurbines;ibi++) {
        fscanf(fd, "%le %le %le %le %le ", &(fsi_wt[ibi].x_c), &(fsi_wt[ibi].y_c), &(fsi_wt[ibi].z_c), &(wtm[ibi].indf_axis), &(wtm[ibi].Tipspeedratio));

        fsi_wt[ibi].x_c=fsi_wt[ibi].x_c/cl;
        fsi_wt[ibi].y_c=fsi_wt[ibi].y_c/cl;
        fsi_wt[ibi].z_c=fsi_wt[ibi].z_c/cl;

        MPI_Bcast(&(fsi_wt[ibi].x_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].y_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].z_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(wtm[ibi].indf_axis), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(wtm[ibi].Tipspeedratio), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        PetscPrintf(PETSC_COMM_WORLD, "The rotating center for %d th turbne %f %f %f \n", ibi, (fsi_wt[ibi].x_c), (fsi_wt[ibi].y_c), (fsi_wt[ibi].z_c));
        PetscPrintf(PETSC_COMM_WORLD, "Induction factor for %d th turbine  %f \n", ibi, (wtm[ibi].indf_axis));
        PetscPrintf(PETSC_COMM_WORLD, "Tipspeedratio for %d th turbine  %f \n", ibi, (wtm[ibi].Tipspeedratio));
      }

      for (ibi=0;ibi<NumberOfTurbines;ibi++) {
        fsi_wt[ibi].x_c0=fsi_wt[ibi].x_c;
        fsi_wt[ibi].y_c0=fsi_wt[ibi].y_c;
        fsi_wt[ibi].z_c0=fsi_wt[ibi].z_c;
      }


      fclose(fd);

      sprintf(str, "%s/AxisDirectionWT.dat", path);
      fd = fopen(str, "r");
      if(!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(1);

      for (ibi=0;ibi<NumberOfTurbines;ibi++) {
        fscanf(fd, "%le %le %le ", &(fsi_wt[ibi].nx_tb), &(fsi_wt[ibi].ny_tb), &(fsi_wt[ibi].nz_tb));

        double rr=sqrt(pow(fsi_wt[ibi].nx_tb,2)+pow(fsi_wt[ibi].ny_tb,2)+pow(fsi_wt[ibi].nz_tb,2));

        fsi_wt[ibi].nx_tb=fsi_wt[ibi].nx_tb/rr;
        fsi_wt[ibi].ny_tb=fsi_wt[ibi].ny_tb/rr;
        fsi_wt[ibi].nz_tb=fsi_wt[ibi].nz_tb/rr;

        MPI_Bcast(&(fsi_wt[ibi].nx_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].ny_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].nz_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        PetscPrintf(PETSC_COMM_WORLD, "The rotating center %f %f %f \n", (fsi_wt[ibi].nx_tb), (fsi_wt[ibi].ny_tb), (fsi_wt[ibi].nz_tb));
      }

      fclose(fd);

    }
    else {

      for (ibi=0;ibi<NumberOfTurbines;ibi++) {
        MPI_Bcast(&(fsi_wt[ibi].x_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].y_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].z_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(wtm[ibi].indf_axis), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(wtm[ibi].Tipspeedratio), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
      }

      for (ibi=0;ibi<NumberOfTurbines;ibi++) {
        fsi_wt[ibi].x_c0=fsi_wt[ibi].x_c;
        fsi_wt[ibi].y_c0=fsi_wt[ibi].y_c;
        fsi_wt[ibi].z_c0=fsi_wt[ibi].z_c;
      }

      for (ibi=0;ibi<NumberOfTurbines;ibi++) {
        MPI_Bcast(&(fsi_wt[ibi].nx_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].ny_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_wt[ibi].nz_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
      }

    }



    PetscBarrier(PETSC_NULL);

    for (i=0;i<NumberOfTurbines;i++) {

      PetscPrintf(PETSC_COMM_WORLD, "Turbines read!\n");

      if (rotor_model == 1) {
        //ACD_read(&wtm[i], i, &fsi_wt[i], 0);
        char fname[80];
        sprintf(fname,"acddata000");
        disk_read_ucd(&wtm[i], i, &fsi_wt[i], 0, fname);
      }

      if (rotor_model == 2 ) {
        //ACD_read(&wtm[i], i, &fsi_wt[i], 0);
        char fname[80];
        sprintf(fname,"acl2data000");
        disk_read_ucd(&wtm[i], i, &fsi_wt[i], 0, fname);
      }

      if (rotor_model == 3 ) ACL_read_ucd(&wtm[i], i, &fsi_wt[i]);

      if ((rotor_model == 3 || rotor_model == 2) && FixTipSpeedRatio) {
        char fname[80];
        sprintf(fname,"Urefdata000");
        disk_read_ucd(&ibm_ACD[i], i, &fsi_wt[i], 1, fname);
        //UrefACD_read(&ibm_ACD[i], i, &fsi_wt[i], 1);
      }

//        		FsiInitialize(0, &fsi_wt[i], i);
    }

    PetscBarrier(PETSC_NULL);
    PetscPrintf(PETSC_COMM_WORLD, "rotor model pre-processing!\n");
    if (rotor_model==1) Pre_process(&(user[0]), wtm, NumberOfTurbines);
    //if (rotor_model==2 || rotor_model==3) Pre_process_wtm(&(user[0]), wtm,fsi_wt,  NumberOfTurbines);
    if (rotor_model==2 || rotor_model==3) Pre_process(&(user[0]), wtm, NumberOfTurbines);

    PetscPrintf(PETSC_COMM_WORLD, "Uref ACD pre-processing!\n");
    if ((rotor_model == 3 || rotor_model == 2) && FixTipSpeedRatio) Pre_process(&(user[0]), ibm_ACD, NumberOfTurbines);

    if (rotor_model == 3 || rotor_model == 2) airfoil_ACL(acl, wtm,  fsi_wt);

    ti = 0;
    if (rstart_flg) ti = tistart;

//		if (rotatewt) {
//    			for (bi=0; bi<block_number; bi++) {
//      			for (ibi=0;ibi<NumberOfTurbines;ibi++) {
//      				Elmt_Move_FSI_ROT(&fsi_wt[ibi], &wtm[ibi], user[bi].dt, ibi);
//			}
//			}
//		}

  }



  if (IB_delta) {

    PetscReal cl = 1.;

    PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &cl, PETSC_NULL);

    if (!my_rank) {

      FILE *fd;
      char str[256];
      sprintf(str, "%s/CenterIBDelta.dat", path);
      fd = fopen(str, "r");
      if(!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(1);

      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {
        fscanf(fd, "%le %le %le %le ", &(fsi_IBDelta[ibi].x_c), &(fsi_IBDelta[ibi].y_c), &(fsi_IBDelta[ibi].z_c), &(ibm_IBDelta[ibi].CD_bluff));

        fsi_IBDelta[ibi].x_c=fsi_IBDelta[ibi].x_c/cl;
        fsi_IBDelta[ibi].y_c=fsi_IBDelta[ibi].y_c/cl;
        fsi_IBDelta[ibi].z_c=fsi_IBDelta[ibi].z_c/cl;

        MPI_Bcast(&(fsi_IBDelta[ibi].x_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].y_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].z_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(ibm_IBDelta[ibi].CD_bluff), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        PetscPrintf(PETSC_COMM_WORLD, "The Locations for %d th  IBDelta %f %f %f \n", ibi, (fsi_IBDelta[ibi].x_c), (fsi_IBDelta[ibi].y_c), (fsi_IBDelta[ibi].z_c));
        PetscPrintf(PETSC_COMM_WORLD, "The drag coefficient for %d th IBDelta body %f \n", ibi, (ibm_IBDelta[ibi].CD_bluff));
      }

      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {
        fsi_IBDelta[ibi].x_c0=fsi_IBDelta[ibi].x_c;
        fsi_IBDelta[ibi].y_c0=fsi_IBDelta[ibi].y_c;
        fsi_IBDelta[ibi].z_c0=fsi_IBDelta[ibi].z_c;
      }


      fclose(fd);

      sprintf(str, "%s/AxisDirectionIBDelta.dat", path);
      fd = fopen(str, "r");
      if(!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(1);

      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {
        fscanf(fd, "%le %le %le ", &(fsi_IBDelta[ibi].nx_tb), &(fsi_IBDelta[ibi].ny_tb), &(fsi_IBDelta[ibi].nz_tb));

        double rr=sqrt(pow(fsi_IBDelta[ibi].nx_tb,2)+pow(fsi_IBDelta[ibi].ny_tb,2)+pow(fsi_IBDelta[ibi].nz_tb,2));

        fsi_IBDelta[ibi].nx_tb=fsi_IBDelta[ibi].nx_tb/rr;
        fsi_IBDelta[ibi].ny_tb=fsi_IBDelta[ibi].ny_tb/rr;
        fsi_IBDelta[ibi].nz_tb=fsi_IBDelta[ibi].nz_tb/rr;

        MPI_Bcast(&(fsi_IBDelta[ibi].nx_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].ny_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].nz_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        PetscPrintf(PETSC_COMM_WORLD, "The directions of IBDelta %f %f %f \n", (fsi_IBDelta[ibi].nx_tb), (fsi_IBDelta[ibi].ny_tb), (fsi_IBDelta[ibi].nz_tb));
      }

      fclose(fd);


    }
    else {

      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {
        MPI_Bcast(&(fsi_IBDelta[ibi].x_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].y_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].z_c), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(ibm_IBDelta[ibi].CD_bluff), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
      }

      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {
        fsi_IBDelta[ibi].x_c0=fsi_IBDelta[ibi].x_c;
        fsi_IBDelta[ibi].y_c0=fsi_IBDelta[ibi].y_c;
        fsi_IBDelta[ibi].z_c0=fsi_IBDelta[ibi].z_c;
      }

      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {
        MPI_Bcast(&(fsi_IBDelta[ibi].nx_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].ny_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&(fsi_IBDelta[ibi].nz_tb), 1, MPIU_REAL, 0, PETSC_COMM_WORLD);
      }

    }


    PetscPrintf(PETSC_COMM_WORLD, "IBDeltas read  11!\n");
    int ipt;
    int NumLoc=NumberOfIBDelta/NumIBPerLoc;
    for (ibi=0;ibi<NumLoc;ibi++)
      for (ipt=0;ipt<NumIBPerLoc;ipt++) {
        int iname=ibi*NumIBPerLoc+ipt;
        char fname[80];
        sprintf(fname,"ibmDelta%3.3d", ipt);
        disk_read_ucd(&ibm_IBDelta[iname], iname, &fsi_IBDelta[iname], 0, fname);
      }

    Pre_process(&(user[0]), ibm_IBDelta, NumberOfIBDelta); // xyang 12-13-2010

    ti = 0;
    if (rstart_flg) ti = tistart;

    for (bi=0; bi<block_number; bi++) {
      for (ibi=0;ibi<NumberOfIBDelta;ibi++) {

//                	Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi);

      }
    }

  }



  // rstart not working now
  level = usermg.mglevels-1;
  user = usermg.mgctx[level].user;
  if (rstart_flg) {
    ti = tistart; tistart++;

    for (bi=0; bi<block_number; bi++) {
      Ucont_P_Binary_Input(&(user[bi]));


      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat);

      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);

      DMGlobalToLocalBegin(user[bi].da, user[bi].P, INSERT_VALUES, user[bi].lP);
      DMGlobalToLocalEnd(user[bi].da, user[bi].P, INSERT_VALUES, user[bi].lP);

      DMGlobalToLocalBegin(user[bi].da, user[bi].Nvert_o, INSERT_VALUES, user[bi].lNvert_o);
      DMGlobalToLocalEnd(user[bi].da, user[bi].Nvert_o, INSERT_VALUES, user[bi].lNvert_o);

//      Contra2Cart(&(user[bi]));

      if (rstart_fsi) {
	for (ibi=0;ibi<NumberOfBodies;ibi++) {

	  if(!rotatefsi) FSI_DATA_Input(&fsi[ibi],ibi);

	  if (movefsi) {
	    Elmt_Move_FSI_TRANS(&fsi[ibi], &ibm[ibi]);
	    for (i=0;i<6;i++){
	      fsi[ibi].S_realm1[i]=fsi[ibi].S_real[i];
	      fsi[ibi].S_real[i]=fsi[ibi].S_new[i];
	    }
	    for (i=0; i<ibm[ibi].n_v; i++) {
	      ibm[ibi].uold[i].x = fsi[ibi].S_real[1];
	      ibm[ibi].uold[i].y = fsi[ibi].S_real[3];
	      ibm[ibi].uold[i].z = fsi[ibi].S_real[5];
	    }
	    for (i=0; i<ibm[ibi].n_v; i++) {
	      ibm[ibi].urm1[i].x = fsi[ibi].S_realm1[1];
	      ibm[ibi].urm1[i].y = fsi[ibi].S_realm1[3];
	      ibm[ibi].urm1[i].z = fsi[ibi].S_realm1[5];
	    }
	  }
	  if (rotatefsi|| MHV) {
	    fsi[ibi].x_c = x_r; //seokkoo
            fsi[ibi].y_c = y_r;
            fsi[ibi].z_c = z_r;

	    if(ibi==0 || ibi<NumberOfRotatingBodies) {
	      Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi);
	    }
	    else {
	      for (i=0; i<ibm[ibi].n_v; i++) {
		ibm[ibi].u[i].x = 0;
		ibm[ibi].u[i].y = 0;
		ibm[ibi].u[i].z = 0;
		ibm[ibi].uold[i] = ibm[ibi].u[i];
		ibm[ibi].urm1[i] = ibm[ibi].u[i];
	      }
	    }

	    // if read ti, then will start for ti+1
	    for (i=0;i<6;i++){
	      fsi[ibi].S_ang_rm1[i]=fsi[ibi].S_ang_r[i];
	      fsi[ibi].S_ang_r[i]=fsi[ibi].S_ang_n[i];
	    }

	    fsi[ibi].F_x_real=fsi[ibi].F_x;
	    fsi[ibi].F_y_real=fsi[ibi].F_y;
	    fsi[ibi].F_z_real=fsi[ibi].F_z;

	    fsi[ibi].M_x_rm3=fsi[ibi].M_x;
	    fsi[ibi].M_y_rm3=fsi[ibi].M_y;
	    fsi[ibi].M_z_rm3=fsi[ibi].M_z;

	    fsi[ibi].M_x_rm2=fsi[ibi].M_x;
	    fsi[ibi].M_y_rm2=fsi[ibi].M_y;
	    fsi[ibi].M_z_rm2=fsi[ibi].M_z;

	    fsi[ibi].M_x_real=fsi[ibi].M_x;
	    fsi[ibi].M_y_real=fsi[ibi].M_y;
	    fsi[ibi].M_z_real=fsi[ibi].M_z;

	    /*
              PetscReal rx,ry,rz;
              for (i=0; i<ibm[ibi].n_v; i++) {
	      rx = ibm[ibi].x_bp[i]-fsi[ibi].x_c;
	      ry = ibm[ibi].y_bp[i]-fsi[ibi].y_c;
	      rz = ibm[ibi].z_bp[i]-fsi[ibi].z_c;


	      ibm[ibi].u[i].x =   ry*fsi[ibi].S_ang_n[5]-fsi[ibi].S_ang_n[3]*rz  ;
	      ibm[ibi].u[i].y =-( rx*fsi[ibi].S_ang_n[5]-fsi[ibi].S_ang_n[1]*rz );
	      ibm[ibi].u[i].z =   rx*fsi[ibi].S_ang_n[3]-fsi[ibi].S_ang_n[1]*ry  ;

	      ibm[ibi].uold[i].x =   ry*fsi[ibi].S_ang_r[5]-fsi[ibi].S_ang_r[3]*rz  ;
	      ibm[ibi].uold[i].y =-( rx*fsi[ibi].S_ang_r[5]-fsi[ibi].S_ang_r[1]*rz );
	      ibm[ibi].uold[i].z =   rx*fsi[ibi].S_ang_r[3]-fsi[ibi].S_ang_r[1]*ry  ;

	      ibm[ibi].urm1[i].x =   ry*fsi[ibi].S_ang_rm1[5]-fsi[ibi].S_ang_rm1[3]*rz  ;
	      ibm[ibi].urm1[i].y =-( rx*fsi[ibi].S_ang_rm1[5]-fsi[ibi].S_ang_rm1[1]*rz );
	      ibm[ibi].urm1[i].z =   rx*fsi[ibi].S_ang_rm1[3]-fsi[ibi].S_ang_rm1[1]*ry  ;

	      }*/
	  }
	}//ibi
      } // if rstart fsi

    }// bi

/*     if (immersed) { */
/*       //fluxin(&(user[bi])); */
/*       //      Elmt_Move(&ibm, &(user[0])); */
/*     } */

  } // if rstart


  for (bi=0; bi<block_number; bi++) {	//111211
    if (levelset) {
      if(ti==tistart && ti==0) {
        PetscPrintf(PETSC_COMM_WORLD, "Initialize water surface elevation!\n");
        Levelset_Function_IC(&user[bi]);
        DMGlobalToLocalBegin(user[bi].da, user[bi].Levelset, INSERT_VALUES, user[bi].lLevelset);
        DMGlobalToLocalEnd(user[bi].da, user[bi].Levelset, INSERT_VALUES, user[bi].lLevelset);
        Compute_Surface_Curv(&user[bi]);
      }

      if (rstart_flg || (ti==tistart && ti==0) ) {
        Compute_Density(&user[bi]);
        if(surface_tension) Compute_Surface_Tension(&user[bi]);
      }
    }
    Contra2Cart(&(user[bi]));
  }

/*   if (flg) { */
/*     ti = tistart; tistart++; */

/*     Ucont_P_Binary_Input(&user); */
/*     DMGlobalToLocalBegin(user.fda, user.Ucont, INSERT_VALUES, user.lUcont); */
/*     DMGlobalToLocalEnd(user.fda, user.Ucont, INSERT_VALUES, user.lUcont); */

/*     DMGlobalToLocalBegin(user.da, user.P, INSERT_VALUES, user.lP); */
/*     DMGlobalToLocalEnd(user.da, user.P, INSERT_VALUES, user.lP); */

/*     DMGlobalToLocalBegin(user.da, user.Nvert_o, INSERT_VALUES, user.lNvert_o); */
/*     DMGlobalToLocalEnd(user.da, user.Nvert_o, INSERT_VALUES, user.lNvert_o); */

/*     Contra2Cart(&user); */
/*     /\*   TecOut(&user); *\/ */
/*     fluxin(); */
/*     Elmt_Move_thin(&ibm, &user); */
/*   } */


// do the search once if elmt is not moving!
  if (immersed) {
    for (level = usermg.mglevels-1; level>=usermg.mglevels-1; level--) {
      user = usermg.mgctx[level].user;
      for (bi=0; bi<block_number; bi++) {
	/*
          if (rotatefsi) {//seokkoo
	  for (ibi=0;ibi<NumberOfBodies;ibi++) {
          fsi[ibi].x_c = 0;
          fsi[ibi].y_c = 0;
          fsi[ibi].z_c = 0;
          Elmt_Move_FSI_ROT(&fsi[ibi], &ibm[ibi], user[bi].dt, ibi);
	  }
	  }*/

	for (ibi=0;ibi<NumberOfBodies;ibi++) {
	  PetscPrintf(PETSC_COMM_WORLD, "IBM_SERA %d \n", ibi);
	  ibm_search_advanced(&(user[bi]), &ibm[ibi], ibi);
          if (sediment) {
            PetscPrintf(PETSC_COMM_WORLD, "Start Connectivity for sediment case %d \n", ibi);
            Connectivity_ib(&(user[bi]), &ibm[ibi]);  // xiaolei add SEDI
            PetscPrintf(PETSC_COMM_WORLD, "End Connectivity for sediment case %d \n", ibi);
	  }
	}
	//if (sediment) preprocess_ib(&(user[bi]), ibm); // xiaolei add SEDI
	MPI_Barrier(PETSC_COMM_WORLD);
	/*for (ibi=0;ibi<NumberOfBodies;ibi++) */{
	  PetscPrintf(PETSC_COMM_WORLD, "IBM_INTP\n");
	  ibm_interpolation_advanced(&user[bi],0);
/* 	  Calc_forces_SI(&fsi[ibi],&(user[bi]),&ibm[ibi], ti, ibi, bi); */

/* 	  fsi[ibi].F_x_old=fsi[ibi].F_x; */
/* 	  fsi[ibi].F_y_old=fsi[ibi].F_y; */
/* 	  fsi[ibi].F_z_old=fsi[ibi].F_z; */

/* 	  fsi[ibi].M_x_old=fsi[ibi].M_x; */
/* 	  fsi[ibi].M_y_old=fsi[ibi].M_y; */
/* 	  fsi[ibi].M_z_old=fsi[ibi].M_z; */
/* 	  if (TwoD && !rstart_fsi)  */
/* 	    //CV_Boundary(&user[bi],&fsi[ibi], ibi); */
/* 	  PetscPrintf(PETSC_COMM_SELF, "IBM_CVBndry\n"); */
/* 	  /\* 	ibm_search(&(user[bi]), &ibm, user[bi].info.mx, user[bi].info.my, user[bi].info.mz, ibm_intp); *\/ */
	}
      }
    }
  }


  // Copy Ucont to Ucont_o for the finest level
  for (bi=0; bi<block_number; bi++) {
    //VecDuplicate(user[bi].Ucont, &(user[bi].Ucont_o));
    ti = 0;
    if (rstart_flg) ti = tistart;
/*
  if(ti==tistart && ti==0 && levelset) {
  Levelset_Function_IC(&user[bi]);
  DMGlobalToLocalBegin(user[bi].da, user[bi].Levelset, INSERT_VALUES, user[bi].lLevelset);
  DMGlobalToLocalEnd(user[bi].da, user[bi].Levelset, INSERT_VALUES, user[bi].lLevelset);
  }
*/
    if(ti==tistart) {
      Set_Nvert_for_Wallfunction(&user[bi]);
      Calc_Inlet_Area(&user[bi]);
    }

    if (ti==0) {
      VecSet(user[bi].Ucont,0.);
      VecSet(user[bi].lUcont,0.);
      VecSet(user[bi].Ucont_o,0.);
      VecSet(user[bi].lUcont_o,0.);
      VecSet(user[bi].Ucat,0.);
      VecSet(user[bi].lUcat,0.);
      VecSet(user[bi].P,0.);
      VecSet(user[bi].lP,0.);

      if(!initialzero) {
        SetInitialGuessToOne(&(user[bi]));
        Scale_InitFlow(&(user[bi])); // xiaolei add
      }

      Contra2Cart(&(user[bi]));
      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat_old);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat_old);
    }

    VecCopy(user[bi].Ucont, user[bi].Ucont_o);
    //VecCopy(user[bi].Ucont, user[bi].Ucont_rm2);	// allocate at init.c
    VecCopy(user[bi].Ucont, user[bi].Ucont_rm1);

    VecCopy(user[bi].Ucat, user[bi].Ucat_o);
    VecCopy(user[bi].P, user[bi].P_o);

    DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont_o, INSERT_VALUES, user[bi].lUcont_o);
    DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont_o, INSERT_VALUES, user[bi].lUcont_o);

    DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont_rm1, INSERT_VALUES, user[bi].lUcont_rm1);
    DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont_rm1, INSERT_VALUES, user[bi].lUcont_rm1);
  }

  /*
    double aa;
    VecNorm(user[0].lUcont, NORM_2, &aa);
    printf("%f\n", aa);
    exit(0);
  */
  MPI_Barrier(PETSC_COMM_WORLD);
  // calculate the forces only on the finest level
/*   for (level = usermg.mglevels-1; level>=usermg.mglevels-1; level--) { */
/*     user = usermg.mgctx[level].user; */
/*     for (bi=0; bi<block_number; bi++) { */
/*       if (immersed) { */
/* 	//fsi_interpolation_coeff(&user[bi], &ibm, fsi.fsi_intp, fsi.elmtinfo, &fsi); 	 */
/*       } */
/*     } */
/*   } */



  tisteps = 1000000;
  PetscOptionsGetInt(PETSC_NULL, "-totalsteps", &tisteps, &flg);

  /*  if (MHV && tistart==21) tistart=0; */

  if (tistart==0) tisteps ++;

/*  put the time accuracy coefficient 1 for the 1st real-time step */
/*   COEF_TIME_ACCURACY=1.; */

/*   PreLoadBegin(PETSC_TRUE, "Load"); */

/* ==================================================================================             */
/*   pysical time Step Loop */
  for (ti = tistart; ti<= tisteps; ti++) {

    PetscPrintf(PETSC_COMM_WORLD, "Time %d\n", ti);
    /*
      if (inletprofile==3) {
      if (MHV && (fsi[1].S_ang_n[0]<0.8*max_angle || fsi[2].S_ang_n[0]>-0.8*max_angle))
      angle=angle+1;
      else
      angle=0.;

      fluxin(&(usermg.mgctx[usermg.mglevels-1].user[0]));
      }
    */
    /* ==================================================================================             */
    /*     Strong-Coupling (SC) Loop */
    PetscReal ts,te,cput;
    PetscGetTime(&ts);

    DoSCLoop= PETSC_TRUE ; itr_sc = 0;
    while (DoSCLoop) {

      itr_sc++;
      PetscPrintf(PETSC_COMM_WORLD, "SC LOOP itr # %d\n", itr_sc);


      /*     Structral Solver! */
      if (immersed/* && (movefsi || rotatefsi)*/)
        Struc_Solver(&usermg, ibm, fsi, itr_sc, &DoSCLoop);
      else
        DoSCLoop = PETSC_FALSE;

/*       /\*     Structral Solver! *\/ */
/*       if (immersed) */
/*       Struc_predictor(&usermg, ibm, fsi, itr_sc,tistart, &DoSCLoop); */
/*       else */
/*       DoSCLoop = PETSC_FALSE; */

      /*     Flow Solver! */
      if(levelset) Calc_Inlet_Area(&(usermg.mgctx[usermg.mglevels-1].user[0]));

      Flow_Solver(&usermg, ibm, fsi, wtm, acl, fsi_wt, ibm_ACD, fsi_IBDelta, ibm_IBDelta);

      if(rotatefsi || movefsi) for (ibi=0;ibi<NumberOfBodies;ibi++) ibm_surface_out_with_pressure(&ibm[ibi], ibi);

    }// End of while SC loop
    /* ==================================================================================             */

/*  put the time accuracy coefficient back to 1.5
    after the 1st real-time step */
/*     COEF_TIME_ACCURACY=1.5; */


/* ==================================================================================             */
/*     Save the old values (at ti) for later */

    level = usermg.mglevels-1;
    user = usermg.mgctx[level].user;
    for (bi=0; bi<block_number; bi++) {

      if (immersed) {
	VecCopy(user[bi].Nvert, user[bi].Nvert_o);
	DMGlobalToLocalBegin(user[bi].da, user[bi].Nvert_o, INSERT_VALUES, user[bi].lNvert_o);
	DMGlobalToLocalEnd(user[bi].da, user[bi].Nvert_o, INSERT_VALUES, user[bi].lNvert_o);
      }

      VecCopy(user[bi].Ucont_o, user[bi].Ucont_rm1);
      VecCopy(user[bi].Ucont, user[bi].Ucont_o);
      VecCopy(user[bi].P, user[bi].P_o);

      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont_o, INSERT_VALUES, user[bi].lUcont_o);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont_o, INSERT_VALUES, user[bi].lUcont_o);

      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont_rm1, INSERT_VALUES, user[bi].lUcont_rm1);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont_rm1, INSERT_VALUES, user[bi].lUcont_rm1);

      //seokkoo
      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat_old);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucat, INSERT_VALUES, user[bi].lUcat_old);


      if (temperature) {

  	PetscPrintf(PETSC_COMM_WORLD, "save old temperature\n");

        VecCopy(user[bi].Tmprt_o, user[bi].Tmprt_rm1);
        VecCopy(user[bi].Tmprt, user[bi].Tmprt_o);

        DMGlobalToLocalBegin(user[bi].da, user[bi].Tmprt_o, INSERT_VALUES, user[bi].lTmprt_o);
        DMGlobalToLocalEnd(user[bi].da, user[bi].Tmprt_o, INSERT_VALUES, user[bi].lTmprt_o);

        DMGlobalToLocalBegin(user[bi].da, user[bi].Tmprt_rm1, INSERT_VALUES, user[bi].lTmprt_rm1);
        DMGlobalToLocalEnd(user[bi].da, user[bi].Tmprt_rm1, INSERT_VALUES, user[bi].lTmprt_rm1);

      }

      if (conv_diff)
      {
  	PetscPrintf(PETSC_COMM_WORLD, "save old concentration\n");
        VecCopy(user[bi].Conc, user[bi].Conc_o);

        DMGlobalToLocalBegin(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
        DMGlobalToLocalEnd(user[bi].da, user[bi].Conc_o, INSERT_VALUES, user[bi].lConc_o);
      }
    }

    if (immersed && (movefsi || rotatefsi || cop || fish || MHV || sediment )){// && ti<tistart+3) {    // SEDI

      for (ibi=0;ibi<NumberOfBodies;ibi++) {


	if (sediment) {
          for(i=0;i<ibm[ibi].n_elmt; i++){
            ibm[ibi].cent_z_old[i] = ibm[ibi].cent_z[i];
          }
	}


        for (i=0; i<ibm[ibi].n_v; i++) {
          ibm[ibi].x_bp_o[i] = ibm[ibi].x_bp[i];
          ibm[ibi].y_bp_o[i] = ibm[ibi].y_bp[i];
          ibm[ibi].z_bp_o[i] = ibm[ibi].z_bp[i];

          ibm[ibi].urm1[i].x = ibm[ibi].uold[i].x;
          ibm[ibi].urm1[i].y = ibm[ibi].uold[i].y;
          ibm[ibi].urm1[i].z = ibm[ibi].uold[i].z;

          ibm[ibi].uold[i].x = ibm[ibi].u[i].x;
          ibm[ibi].uold[i].y = ibm[ibi].u[i].y;
          ibm[ibi].uold[i].z = ibm[ibi].u[i].z;
        }

        for (i=0;i<6;i++){
          fsi[ibi].S_realm1[i]=fsi[ibi].S_real[i];
          fsi[ibi].S_real[i]=fsi[ibi].S_new[i];

          fsi[ibi].S_ang_rm1[i]=fsi[ibi].S_ang_r[i];
          fsi[ibi].S_ang_r[i]=fsi[ibi].S_ang_n[i];
        }

        fsi[ibi].F_x_real=fsi[ibi].F_x;
        fsi[ibi].F_y_real=fsi[ibi].F_y;
        fsi[ibi].F_z_real=fsi[ibi].F_z;

        fsi[ibi].M_x_rm3=fsi[ibi].M_x_rm2;
        fsi[ibi].M_y_rm3=fsi[ibi].M_y_rm2;
        fsi[ibi].M_z_rm3=fsi[ibi].M_z_rm2;

        fsi[ibi].M_x_rm2=fsi[ibi].M_x_real;
        fsi[ibi].M_y_rm2=fsi[ibi].M_y_real;
        fsi[ibi].M_z_rm2=fsi[ibi].M_z_real;

        fsi[ibi].M_x_real=fsi[ibi].M_x;
        fsi[ibi].M_y_real=fsi[ibi].M_y;
        fsi[ibi].M_z_real=fsi[ibi].M_z;


/*       fsi[ibi].F_x_old=fsi[ibi].F_x; */
/*       fsi[ibi].F_y_old=fsi[ibi].F_y; */
/*       fsi[ibi].F_z_old=fsi[ibi].F_z; */

/*       fsi[ibi].M_x_old=fsi[ibi].M_x; */
/*       fsi[ibi].M_y_old=fsi[ibi].M_y; */
/*       fsi[ibi].M_z_old=fsi[ibi].M_z; */

      } //ibi
    }


    PetscGetTime(&te);
    cput=te-ts;
    if (!my_rank) {
      FILE *f;
      char filen[80];
      sprintf(filen, "%s/Converge_dU", path);
      f = fopen(filen, "a");
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d(total)    %.2e(s)\n", ti, cput);
      fclose(f);
    }


/* ==================================================================================             */

////////////////////////////////////---------------------------
/*     if (ti == (ti/100)*100) */
/*       Ucont_P_Binary_Output(&user); */

  } // ti (physical time) loop
/* ==================================================================================             */
  PetscPrintf(PETSC_COMM_WORLD, "\n\n ******* Finished computation ti=%d ******* \n\n", ti);

  MG_Finalize(&usermg);
  PetscFinalize();

/* ==================================================================================             */
  return(0);

}
