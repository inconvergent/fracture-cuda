#define THREADS _THREADS_
#define PROX _PROX_

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__device__ int get_fow_items(
    const int nz,
    const int zx,
    const int zy,
    const int *zone_num,
    const int *zone_node,
    const int zone_leap,
    const int aa,
    const int ff,
    const int ii,
    float *tmp,
    const float *xy,
    const float *dxy,
    const float fow_dot,
    const float fow_dst,
    int *proximity
    ){
  int zk;
  int jj;
  float dd;
  float dt;
  float jdx;
  float jdy;
  float nrm;

  float x = xy[ii];
  float y = xy[ii+1];
  float fdx = dxy[ff];
  float fdy = dxy[ff+1];

  int count = 0;
  for (int a=max(zx-1,0);a<min(zx+2,nz);a++){
    for (int b=max(zy-1,0);b<min(zy+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){
        jj = 2*zone_node[zk*zone_leap+k];

        if (jj == ii){
          continue;
        }

        jdx = xy[jj]-x;
        jdy = xy[jj+1]-y;
        nrm = sqrt(jdx*jdx+jdy*jdy);

        dt = (fdx*jdx + fdy*jdy)/nrm;
        dd = dist(xy, xy, jj, ii);

        if (dd>0.0f && dd<fow_dst && dt>fow_dot){
          proximity[count] = jj/2;
          count += 1;
        }
        else{
          tmp[ii] = dd;
          tmp[ii+1] = dt;
        }
      }
    }
  }


  return count;
}

__global__ void calc_stp(
    const int nz,
    const int zone_leap,
    const int num,
    const int fnum,
    const int anum,
    const float frac_dot,
    const float frac_dst,
    const float frac_stp,
    const int *visited,
    const int *fid_node,
    const int *active,
    float *tmp,
    const float *xy,
    const float *dxy,
    float *ndxy,
    const int *zone_num,
    const int *zone_node
    ){
  const int a = blockIdx.x*THREADS + threadIdx.x;

  if (a>=anum){
    return;
  }

  const int aa = 2*a;

  const int ff = 2*active[a];
  const int ii = 2*fid_node[ff+1];

  const int zx = (int) floor(xy[ii]*nz);
  const int zy = (int) floor(xy[ii+1]*nz);

  tmp[aa] = 21.0f;
  tmp[aa+1] = 22.0f;

  int proximity[PROX];
  int fow_count = get_fow_items(
      nz,
      zx,
      zy,
      zone_num,
      zone_node,
      zone_leap,
      aa,
      ff,
      ii,
      tmp,
      xy,
      dxy,
      frac_dot,
      frac_dst,
      proximity
      );

  if (fow_count<1){
    ndxy[aa] = -100.0f;
    ndxy[aa+1] = -100.0f;
    return;
  }

  float mx = 0.0f;
  float my = 0.0f;
  float nrm;

  int jj;
  for (int k=0;k<fow_count;k++){
    jj = 2*proximity[k];
    mx += xy[jj];
    my += xy[jj+1];
  }

  mx /= (float)fow_count;
  my /= (float)fow_count;
  mx -= xy[ii];
  my -= xy[ii+1];
  nrm = sqrt(mx*mx + my*my);
  mx /= nrm;
  my /= nrm;

  ndxy[aa] = mx;
  ndxy[aa+1] = my;
}

