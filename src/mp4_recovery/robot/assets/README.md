# Robot assets

The Mini Pupper v4 (SKU2) model, from the [mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim)
model build. `pupper.xml` is a generated MJCF (12 hinges, 4 knee tendons, 12 position actuators, IMU site, `stand`
keyframe, ghost copy `g_*`).

| File | What it is |
|---|---|
| `sku_2/pupper.xml` | The model: 59.75 mm shins, mesh feet (19.3 x 19.6 x 6.0 mm) |
| `sku_2/scene.xml` | `pupper.xml` plus floor and lights, for the viewer |
| `sku_2/servo_map.json` | Firmware servo angles to model joint angles |
| `sku_2/parts/` | The meshes `pupper.xml` references, without the visual-only accessories (`acc_*`, `base_screen_face.obj`, `screen_face.png`: lidar, Raspberry Pi, display) |

`pupper.xml` carries a 360 g mass budget. `mp4_recovery.robot.sku2.get_spec` reads it and, without editing the file, removes
the ghost, keyframe, XML actuators and sensor and the accessories' geoms; adds the IMU (`imu_ang_vel`, `imu_lin_vel`,
`imu_lin_acc` on site `imu`); makes the feet a sphere (`foot_radius_m`, default 5.5 mm, its bottom at the mesh foot's lowest
point at the stand) or keeps the mesh; and applies a mass file from `mp4_recovery/data/` (default `sku2_masses.json`, the
weighed robot, 630.5 g). The stand comes from `data/sku2_contract.json`, not the keyframe. A policy's metadata records the
model file's sha256 (`robot.sku2.legs_meta`), and the robot's runner refuses a policy trained on another.
