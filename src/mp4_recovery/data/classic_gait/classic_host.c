/* classic_host.c: the official firmware's Stanford trot and IK (firmware-dev main/robot/stanford_gait.cc and
 * stanford_kinematics.cc, vendored unchanged beside this file) on the host, for many robots at once.
 *
 * The gait keeps its state in file-static variables (sg_ticks, foot_x/y/z). Including its source here puts those in
 * this translation unit, so a batch call loads each robot's state into them, runs the firmware's own
 * stanford_gait_step(), and stores the state back: every robot runs the unmodified firmware code with its own state.
 *
 * Ticks: robot.cc set_servo_angle(i, sdeg[i] + s_offset[i]) (sg_write), i.e.
 *     sig = 511 + (int)(deg * (10240 / scale_dd)), clamped to 0..1023,
 * with servo_ticks_per_deg() = 10240.0f / scale_dd evaluated first, as the firmware does. Built by classic_gait.py
 * with gcc -O2 -ffp-contract=off (no fused multiply-add, as the ESP32's single-precision FPU code without -ffast-math).
 */
#include "stanford_gait.cc"
#include "stanford_kinematics.h"

/* state: per robot 12 floats (foot_x[4], foot_y[4], foot_z[4], Stanford signs) and one tick counter */
static void load(const float *s, unsigned long t)
{
    for (int i = 0; i < 4; i++) { foot_x[i] = s[i]; foot_y[i] = s[4 + i]; foot_z[i] = s[8 + i]; }
    sg_ticks = t;
}

static void save(float *s, unsigned long *t)
{
    for (int i = 0; i < 4; i++) { s[i] = foot_x[i]; s[4 + i] = foot_y[i]; s[8 + i] = foot_z[i]; }
    *t = sg_ticks;
}

/* stanford_gait_reset(height_mm) for the robots with mask[r] != 0 (mask NULL: all) */
void ch_reset(int n, const int *mask, float height_mm, float *state, unsigned long *ticks)
{
    for (int r = 0; r < n; r++) {
        if (mask && !mask[r]) continue;
        stanford_gait_reset(height_mm);
        save(state + 12 * r, ticks + r);
    }
}

/* one stanford_gait_step(vx, vy, wz, height, clearance, feet) per robot with mask[r] != 0 (mask NULL: all);
 * cmd: per robot vx mm/s, vy mm/s, wz rad/s; feet: per robot FR, FL, RR, RL x (x, y, z) mm (sg_foot_t) */
void ch_step(int n, const int *mask, const float *cmd, float height_mm, float clearance_mm, float *state,
             unsigned long *ticks, float *feet)
{
    sg_foot_t f[4];
    for (int r = 0; r < n; r++) {
        if (mask && !mask[r]) continue;
        load(state + 12 * r, ticks[r]);
        stanford_gait_step(cmd[3 * r], cmd[3 * r + 1], cmd[3 * r + 2], height_mm, clearance_mm, f);
        save(state + 12 * r, ticks + r);
        for (int i = 0; i < 4; i++) {
            feet[12 * r + 3 * i] = f[i].x;
            feet[12 * r + 3 * i + 1] = f[i].y;
            feet[12 * r + 3 * i + 2] = f[i].z;
        }
    }
}

/* the IK's servo degrees (before the offsets; servo order FR, FL, RR, RL x abduction, hip, knee) and the firmware's
 * ticks for each robot's four feet; off12: the calibration offsets (deg, servo order); scale_dd: the boards' range.
 * mul_first 0: firmware-dev robot.cc, deg * (10240 / scale_dd); 1: the earlier firmware's servo_write and our
 * contract.py, (deg * 10240) / scale_dd (float32 rounding can put a value on the other side of a tick boundary). */
void ch_servo(int n, const float *feet, const float *off12, float scale_dd, int mul_first, float *sdeg12,
              int *ticks12)
{
    sg_foot_t f[4];
    float sd[13];
    float tpd = 10240.0f / scale_dd;              /* servo_ticks_per_deg() */
    for (int r = 0; r < n; r++) {
        for (int i = 0; i < 4; i++) {
            f[i].x = feet[12 * r + 3 * i];
            f[i].y = feet[12 * r + 3 * i + 1];
            f[i].z = feet[12 * r + 3 * i + 2];
        }
        stanford_kinematics_servo_deg(f, sd);
        for (int i = 1; i <= 12; i++) {
            float deg = sd[i] + off12[i - 1];
            int sig = 511 + (mul_first ? (int)(deg * 10240.0f / scale_dd) : (int)(deg * tpd));
            if (sig < 0) sig = 0;
            if (sig > 1023) sig = 1023;
            sdeg12[12 * r + i - 1] = sd[i];
            ticks12[12 * r + i - 1] = sig;
        }
    }
}
