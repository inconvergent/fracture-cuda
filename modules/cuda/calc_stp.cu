#define THREADS _THREADS_
#define PROX _PROX_
#define BAD -100.0f

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}
__device__ float pdist(
    const float ax,
    const float ay,
    const float bx,
    const float by
    ){
    return sqrt(powf(ax-bx, 2.0f) + powf(ay-by, 2.0f));
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
      }
    }
  }

  return count;
}

__device__ int get_neigh_items(
    const int nz,
    const int zx,
    const int zy,
    const int *zone_num,
    const int *zone_node,
    const int zone_leap,
    const int ii,
    const float *xy,
    const float dst,
    int *proximity
    ){
  int zk;
  int jj;
  float dd;

  int count = 0;
  for (int a=max(zx-1,0);a<min(zx+2,nz);a++){
    for (int b=max(zy-1,0);b<min(zy+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){
        jj = 2*zone_node[zk*zone_leap+k];

        if (jj == ii){
          continue;
        }

        dd = dist(xy, xy, jj, ii);
        if (dd>0.0f && dd<dst){
          proximity[count] = jj/2;
          count += 1;
        }
      }
    }
  }

  return count;
}

__device__ bool new_position_is_ok(
      const float stp,
      const int aa,
      const int ii,
      const float mx,
      const float my,
      const int *visited,
      const float *xy,
      float *tmp,
      const int *proximity,
      const int proximity_count
      ){
  int pp;

  float px;
  float py;

  const float x = xy[ii];
  const float y = xy[ii+1];
  const float sx = x + mx*stp;
  const float sy = y + my*stp;

  for (int l=0;l<proximity_count;l++){
    pp = 2*proximity[l];
    if (pp==ii){
      continue;
    }
    if (visited[pp/2]<0){
      continue;
    }
    px = xy[pp];
    py = xy[pp+1];
    if (stp>max(
          pdist(px, py, x, y),
          pdist(px, py, sx, sy)
          )){
      tmp[aa] = (float)pp*0.5f;
      return false;
    }
  }
  return true;
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
      xy,
      dxy,
      frac_dot,
      frac_dst,
      proximity
      );

  if (fow_count<1){
    ndxy[aa] = BAD;
    ndxy[aa+1] = BAD;
    tmp[aa+1] = 88.0f;
    return;
  }

  float mx = 0.0f;
  float my = 0.0f;
  int jj;
  for (int k=0;k<fow_count;k++){
    jj = 2*proximity[k];
    mx += xy[jj];
    my += xy[jj+1];
  }

  float stpx = mx/((float)fow_count) - xy[ii];
  float stpy = my/((float)fow_count) - xy[ii+1];
  float nrm = sqrt(stpx*stpx + stpy*stpy);
  stpx /= nrm;
  stpy /= nrm;

  const int neigh_count = get_neigh_items(
      nz,
      zx,
      zy,
      zone_num,
      zone_node,
      zone_leap,
      ii,
      xy,
      frac_dst*0.5,
      proximity
      );

  bool res = new_position_is_ok(
      frac_stp,
      aa,
      ii,
      stpx,
      stpy,
      visited,
      xy,
      tmp,
      proximity,
      neigh_count
      );

  if (res){
    ndxy[aa] = stpx;
    ndxy[aa+1] = stpy;
  }
  else{
    ndxy[aa] = BAD;
    ndxy[aa+1] = BAD;
    tmp[aa+1] = 77.0f;
  }
}

