# Expected JSON format for the measurement data

```
{
    "track_uuid": "9d2b4e62-9706-11e7-a379-a7739af88881",
    "t": 1503882379340,
    "att_pitch": 0.986945,
    "att_roll": -0.015609,
    "att_yaw": 0.043594,
    "rot_rate_x": 0.035436,
    "rot_rate_y": -0.011044,
    "rot_rate_z": 0.0009,
    "g_x": -0.008604,
    "g_y": -0.834346,
    "g_z": -0.551174,
    "user_a_x": 0.019087,
    "user_a_y": -0.001607,
    "user_a_z": -0.006016,
    "m_x": 0,
    "m_y": 0,
    "m_z": 0,
    "lat": 31.315337,
    "long": 121.447429,
    "alt": 5.796448,
    "course": 0
}
```

Where each field has the following meaning and format:

- track_uuid: uuid identifier of this track(string)
- t: unix timestamp(milliseconds)
- att_pitch: attitude_pitch(radians)
- att_roll: attitude_roll(radians)
- att_yaw: attitude_yaw(radians)
- rot_rate_x: rotation_rate_x(radians/s)
- rot_rate_y: rotation_rate_y(radians/s)
- rot_rate_z: rotation_rate_z(radians/s)
- g_x: gravity_x(G)
- g_y: gravity_y(G)
- g_z: gravity_z(G)
- user_a_x: user_acc_x(G)
- user_a_y: user_acc_y(G)
- user_a_z: user_acc_z(G)
- m_x: magnetic_field_x(microteslas)
- m_y: magnetic_field_y(microteslas)
- m_z: magnetic_field_z(microteslas)
- lat: latitude(degree)
- long: longitude(degree)
- alt: altitude(meter)
- speed: speed(m/s)
- course: course(degree)