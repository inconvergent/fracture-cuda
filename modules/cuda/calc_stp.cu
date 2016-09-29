#define THREADS _THREADS_
#define PROX _PROX_

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__device__ int get_fow_items(
    const int nz,
    const int zi,
    const int zj,
    const int *zone_num,
    const int *zone_node,
    const int zone_leap,
    const int ii,
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
  float fdx;
  float fdy;
  float nrm;

  int count = 0;

  for (int a=max(zi-1,0);a<min(zi+2,nz);a++){
    for (int b=max(zj-1,0);b<min(zj+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){
        jj = 2*zone_node[zk*zone_leap+k];

        if (jj == ii){
          continue;
        }

        fdx = xy[jj]-xy[ii];
        fdy = xy[jj+1]-xy[ii+1];
        nrm = sqrt(fdx*fdx+fdy*fdy);

        dt = (dxy[jj]*fdx + dxy[jj+1]*fdy)/nrm;
        dd = dist(xy, xy, jj, ii);

        if (dd<fow_dst && dt>fow_dot){
          proximity[count] = jj/2;
          count += 1;
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
    const int *fid_node,
    const int *active,
    const float *xy,
    const float *dxy,
    float *ndxy,
    const int *zone_num,
    const int *zone_node
    ){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i>=anum){
    return;
  }

  const int ii = 2*i;
  const int zi = (int) floor(xy[ii]*nz);
  const int zj = (int) floor(xy[ii+1]*nz);

  const int ff = 2*fid_node[2*active[i]+1];
  /*int ii = 2*fid_node[ff+1];*/

  int proximity[PROX];
  int fow_count = get_fow_items(
      nz,
      zi,
      zj,
      zone_num,
      zone_node,
      zone_leap,
      ff,
      xy,
      dxy,
      frac_dot,
      frac_dst,
      proximity
      );

  if (fow_count<1){
    ndxy[ii] = -100.0f;
    ndxy[ii+1] = -100.0f;
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
  nrm = sqrt(mx*mx+my*my);
  mx /= nrm;
  my /= nrm;

  ndxy[ii] = mx;
  ndxy[ii+1] = my;

  /*  dd = sqrt(powf(dx, 2.0f) + powf(dy, 2.0f));*/
  /**/
  /*  if (dd<=0.0f){*/
  /*    continue;*/
  /*  }*/
  /**/
  /*  rel_neigh = true;*/
  /*  for (int l=0;l<cand_count;l++){*/
  /*    aa = 2*proximity[l];*/
  /*    if (dd>link_ignore_rad){*/
  /*      linked = false;*/
  /*      break;*/
  /*    }*/
  /*    if (dd>max(dist(xy, xy, aa, ii), dist(xy, xy, jj, aa))){*/
  /*      linked = false;*/
  /*      break;*/
  /*    }*/
  /*  }*/
}

