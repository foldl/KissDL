#include <stdint.h>
#include <math.h>

#define CLIP(v, a, b) ((v) < (a) ? (a) : ((v) > (b) ? b : v))

#define HWD_OFFSET(shape, h, w, d) (h * shape[1] * shape[2] + w * shape[2] + d)
#define NHWD_OFFSET(shape, n, h, w, d) (n * shape[1] * shape[2] * shape[3] + h * shape[2] * shape[3] + w * shape[3] + d)

#ifdef __GNUC__

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#else

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#endif

void model_run(void);

