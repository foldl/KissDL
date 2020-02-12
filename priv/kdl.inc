#include <stdint.h>
#include <math.h>

#define CLIP(v, a, b) ((v) < (a) ? (a) : ((v) > (b) ? b : v))

#define HWD_OFFSET(shape, h, w, d) (h * shape[1] * shape[2] + w * shape[2] + d)
#define NHWD_OFFSET(shape, n, h, w, d) (n * shape[1] * shape[2] * shape[3] + h * shape[2] * shape[3] + w * shape[3] + d)

//
void model_run(void);

