#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// 6-DOF ROBOT + GRIPPER
// Arduino UNO + PCA9685 16-Channel I2C Servo Driver
//
// ROS2 UART COMMANDS:
//
// J,0,ANGLE  -> BASE
// J,1,ANGLE  -> J1  (CH1 + CH2 together)
// J,2,ANGLE  -> J2  (CH3)
// J,3,ANGLE  -> J3  (CH4)
// J,4,ANGLE  -> J4  (CH5)
// J,5,ANGLE  -> J5  (CH6)
// J,6,ANGLE  -> J6  (CH7)
// J,7,ANGLE  -> GRIPPER (CH8)
//
// S,SPEED
// HOME
// STOP
// P
//
// ============================================================


// ============================================================
// PCA9685
// ============================================================

#define PCA9685_ADDR 0x40

Adafruit_PWMServoDriver pwm(PCA9685_ADDR);


// ============================================================
// PHYSICAL PCA9685 CHANNELS
// ============================================================

// Base
#define CH_BASE       0

// J1 - TWO PHYSICAL SERVOS
#define CH_J1_A       1
#define CH_J1_B       2

// Remaining joints
#define CH_J2         3
#define CH_J3         4
#define CH_J4         5
#define CH_J5         6
#define CH_J6         7

// Gripper
#define CH_GRIPPER    8

#define NUM_SERVOS    9


// ============================================================
// SERVO PWM SETTINGS
// ============================================================

#define SERVO_FREQ    50

#define MIN_US        500
#define MAX_US        2500
#define CENTER_US     1500


// ============================================================
// SERIAL SETTINGS
// ============================================================

#define SERIAL_BAUD 115200

#define SERIAL_BUFFER_SIZE 64

char serialBuffer[SERIAL_BUFFER_SIZE];

uint8_t serialIndex = 0;


// ============================================================
// MOTION SETTINGS
// ============================================================

#define MOTION_UPDATE_MS 5

unsigned long lastMotionUpdate = 0;


// ============================================================
// SPEED SETTINGS
// ============================================================
//
// S,1  -> 50 deg/s
// S,2  -> 100 deg/s
// ...
// S,10 -> 500 deg/s
//
// ============================================================

int motionSpeed = 5;

float speedDegPerSec[11] =
{
  0.0,
  50.0,
  100.0,
  150.0,
  200.0,
  250.0,
  300.0,
  350.0,
  400.0,
  450.0,
  500.0
};


// ============================================================
// CURRENT SERVO ANGLES
// ============================================================

float currentAngle[NUM_SERVOS] =
{
  0.0,  // CH0
  0.0,  // CH1
  0.0,  // CH2
  0.0,  // CH3
  0.0,  // CH4
  0.0,  // CH5
  0.0,  // CH6
  0.0,  // CH7
  0.0   // CH8
};


// ============================================================
// TARGET SERVO ANGLES
// ============================================================

float targetAngle[NUM_SERVOS] =
{
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0
};


// ============================================================
// CURRENT PWM
// ============================================================

int currentPulse[NUM_SERVOS] =
{
  CENTER_US,
  CENTER_US,
  CENTER_US,
  CENTER_US,
  CENTER_US,
  CENTER_US,
  CENTER_US,
  CENTER_US,
  CENTER_US
};


// ============================================================
// MOTION ENABLE
// ============================================================

bool motionEnabled = true;


// ============================================================
// ANGLE → PWM MICROSECONDS
// ============================================================
//
// -180° -> 500 us
//    0° -> 1500 us
// +180° -> 2500 us
//
// ALL SERVOS INCLUDING GRIPPER
// USE THE SAME RANGE.
//
// ============================================================

int angleToPulse(float angle)
{
  angle = constrain(
    angle,
    -180.0,
    180.0
  );

  float pulse =
    CENTER_US +
    (angle * 1000.0 / 180.0);

  pulse = constrain(
    pulse,
    MIN_US,
    MAX_US
  );

  return (int)round(pulse);
}


// ============================================================
// MICROSECONDS → PCA9685 COUNTS
// ============================================================

uint16_t pulseToCounts(int pulse_us)
{
  // 50 Hz
  // Period = 20,000 us
  // PCA9685 = 4096 counts

  return (uint16_t)(
    ((long)pulse_us * 4096L) / 20000L
  );
}


// ============================================================
// WRITE SERVO PWM
// ============================================================

void writeServoPWM(
  uint8_t channel,
  float angle
)
{
  if (channel >= NUM_SERVOS)
    return;

  int pulse =
    angleToPulse(angle);

  uint16_t counts =
    pulseToCounts(pulse);

  pwm.setPWM(
    channel,
    0,
    counts
  );

  currentPulse[channel] =
    pulse;
}


// ============================================================
// SET PHYSICAL SERVO TARGET
// ============================================================

void setServoTarget(
  uint8_t channel,
  float angle
)
{
  if (channel >= NUM_SERVOS)
    return;

  angle = constrain(
    angle,
    -180.0,
    180.0
  );

  targetAngle[channel] =
    angle;

  motionEnabled = true;
}


// ============================================================
// SET J1 TARGET
// ============================================================
//
// J1 has TWO physical servos:
//
// CH1 = J1-A
// CH2 = J1-B
//
// Both always receive exactly the same target.
//
// ============================================================

void setJ1Target(float angle)
{
  angle = constrain(
    angle,
    -180.0,
    180.0
  );

  targetAngle[CH_J1_A] = angle;
  targetAngle[CH_J1_B] = -angle;

  motionEnabled = true;
}


// ============================================================
// UPDATE MOTION
// ============================================================
//
// Non-blocking motion.
//
// Every 5 ms the current angle moves toward target.
//
// ============================================================

void updateMotion()
{
  if (!motionEnabled)
    return;

  unsigned long now =
    millis();

  if (
    now - lastMotionUpdate <
    MOTION_UPDATE_MS
  )
  {
    return;
  }

  lastMotionUpdate =
    now;

  // 5 ms = 0.005 seconds

  float dt =
    MOTION_UPDATE_MS / 1000.0;

  float step =
    speedDegPerSec[motionSpeed] *
    dt;


  for (
    uint8_t channel = 0;
    channel < NUM_SERVOS;
    channel++
  )
  {
    float difference =
      targetAngle[channel] -
      currentAngle[channel];


    // --------------------------------------------------------
    // Already at target
    // --------------------------------------------------------

    if (
      fabs(difference) < 0.01
    )
    {
      currentAngle[channel] =
        targetAngle[channel];

      writeServoPWM(
        channel,
        currentAngle[channel]
      );

      continue;
    }


    // --------------------------------------------------------
    // Positive direction
    // --------------------------------------------------------

    if (difference > 0.0)
    {
      if (difference <= step)
      {
        currentAngle[channel] =
          targetAngle[channel];
      }
      else
      {
        currentAngle[channel] +=
          step;
      }
    }


    // --------------------------------------------------------
    // Negative direction
    // --------------------------------------------------------

    else
    {
      if (-difference <= step)
      {
        currentAngle[channel] =
          targetAngle[channel];
      }
      else
      {
        currentAngle[channel] -=
          step;
      }
    }


    // --------------------------------------------------------
    // ALL SERVOS: -180° to +180°
    // INCLUDING GRIPPER
    // --------------------------------------------------------

    currentAngle[channel] =
      constrain(
        currentAngle[channel],
        -180.0,
        180.0
      );


    // --------------------------------------------------------
    // Send PWM
    // --------------------------------------------------------

    writeServoPWM(
      channel,
      currentAngle[channel]
    );
  }
}


// ============================================================
// HOME
// ============================================================

void homePosition()
{
  for (
    uint8_t channel = 0;
    channel < NUM_SERVOS;
    channel++
  )
  {
    targetAngle[channel] =
      0.0;
  }

  motionEnabled = true;

  Serial.println(
    "HOME_COMMAND"
  );
}


// ============================================================
// STOP
// ============================================================

void stopMotion()
{
  // Freeze at current position

  for (
    uint8_t channel = 0;
    channel < NUM_SERVOS;
    channel++
  )
  {
    targetAngle[channel] =
      currentAngle[channel];
  }

  motionEnabled = false;

  Serial.println(
    "STOPPED"
  );
}


// ============================================================
// PRINT POSITIONS
// ============================================================

void printPositions()
{
  Serial.println();
  Serial.println(
    "POSITIONS"
  );

  for (
    uint8_t channel = 0;
    channel < NUM_SERVOS;
    channel++
  )
  {
    Serial.print("CH");
    Serial.print(channel);

    Serial.print(" CURRENT=");
    Serial.print(
      currentAngle[channel],
      2
    );

    Serial.print(" TARGET=");
    Serial.print(
      targetAngle[channel],
      2
    );

    Serial.print(" PWM=");
    Serial.println(
      currentPulse[channel]
    );
  }

  Serial.print(
    "SPEED="
  );

  Serial.println(
    motionSpeed
  );

  Serial.println(
    "POSITIONS_END"
  );
}


// ============================================================
// SPEED COMMAND
// ============================================================
//
// S,1
// S,2
// ...
// S,10
//
// ============================================================

void processSpeedCommand(
  char *command
)
{
  char *comma =
    strchr(
      command,
      ','
    );

  if (comma == NULL)
  {
    Serial.println(
      "ERROR,BAD_SPEED_FORMAT"
    );

    return;
  }

  int speed =
    atoi(
      comma + 1
    );

  speed =
    constrain(
      speed,
      1,
      10
    );

  motionSpeed =
    speed;

  Serial.print(
    "SPEED,"
  );

  Serial.println(
    motionSpeed
  );
}


// ============================================================
// JOINT COMMAND
// ============================================================
//
// IMPORTANT:
//
// ROS2 LOGICAL JOINT NUMBER
// IS NOT THE SAME AS PCA9685 CHANNEL.
//
//
// J,0,ANGLE
//     ↓
// CH0 BASE
//
//
// J,1,ANGLE
//     ↓
// CH1 + CH2
// J1 TWO SERVOS
//
//
// J,2,ANGLE
//     ↓
// CH3
// J2
//
//
// J,3,ANGLE
//     ↓
// CH4
// J3
//
//
// J,4,ANGLE
//     ↓
// CH5
// J4
//
//
// J,5,ANGLE
//     ↓
// CH6
// J5
//
//
// J,6,ANGLE
//     ↓
// CH7
// J6
//
//
// J,7,ANGLE
//     ↓
// CH8
// GRIPPER
//
// ============================================================

void processJointCommand(
  char *command
)
{
  char *firstComma =
    strchr(
      command,
      ','
    );

  if (
    firstComma == NULL
  )
  {
    return;
  }


  char *secondComma =
    strchr(
      firstComma + 1,
      ','
    );

  if (
    secondComma == NULL
  )
  {
    return;
  }


  // Separate logical joint number
  // from angle

  *secondComma = '\0';


  int logicalJoint =
    atoi(
      firstComma + 1
    );


  float angle =
    atof(
      secondComma + 1
    );


  // ----------------------------------------------------------
  // Logical joint validation
  //
  // 0 = Base
  // 1 = J1
  // 2 = J2
  // 3 = J3
  // 4 = J4
  // 5 = J5
  // 6 = J6
  // 7 = Gripper
  // ----------------------------------------------------------

  if (
    logicalJoint < 0 ||
    logicalJoint > 7
  )
  {
    return;
  }


  angle = constrain(
    angle,
    -180.0,
    180.0
  );


  // ==========================================================
  // LOGICAL JOINT 0
  // BASE
  // ==========================================================

  if (logicalJoint == 0)
  {
    setServoTarget(
      CH_BASE,
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 1
  // J1
  //
  // CH1 + CH2
  // ==========================================================

  if (logicalJoint == 1)
  {
    setJ1Target(
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 2
  // J2 -> CH3
  // ==========================================================

  if (logicalJoint == 2)
  {
    setServoTarget(
      CH_J2,
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 3
  // J3 -> CH4
  // ==========================================================

  if (logicalJoint == 3)
  {
    setServoTarget(
      CH_J3,
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 4
  // J4 -> CH5
  // ==========================================================

  if (logicalJoint == 4)
  {
    setServoTarget(
      CH_J4,
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 5
  // J5 -> CH6
  // ==========================================================

  if (logicalJoint == 5)
  {
    setServoTarget(
      CH_J5,
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 6
  // J6 -> CH7
  // ==========================================================

  if (logicalJoint == 6)
  {
    setServoTarget(
      CH_J6,
      angle
    );

    return;
  }


  // ==========================================================
  // LOGICAL JOINT 7
  // GRIPPER -> CH8
  //
  // FULL -180° TO +180°
  // ==========================================================

  if (logicalJoint == 7)
  {
    setServoTarget(
      CH_GRIPPER,
      angle
    );

    return;
  }
}


// ============================================================
// PROCESS COMMAND
// ============================================================

void processCommand(
  char *command
)
{
  // ----------------------------------------------------------
  // Remove leading spaces
  // ----------------------------------------------------------

  while (
    *command == ' ' ||
    *command == '\t'
  )
  {
    command++;
  }


  // ----------------------------------------------------------
  // Remove trailing spaces
  // ----------------------------------------------------------

  int len =
    strlen(command);

  while (
    len > 0 &&
    (
      command[len - 1] == ' ' ||
      command[len - 1] == '\t'
    )
  )
  {
    command[len - 1] =
      '\0';

    len--;
  }


  if (len == 0)
    return;


  // ----------------------------------------------------------
  // HOME
  // ----------------------------------------------------------

  if (
    strcmp(command, "HOME") == 0 ||
    strcmp(command, "H") == 0
  )
  {
    homePosition();
    return;
  }


  // ----------------------------------------------------------
  // STOP
  // ----------------------------------------------------------

  if (
    strcmp(command, "STOP") == 0
  )
  {
    stopMotion();
    return;
  }


  // ----------------------------------------------------------
  // PRINT
  // ----------------------------------------------------------

  if (
    strcmp(command, "P") == 0
  )
  {
    printPositions();
    return;
  }


  // ----------------------------------------------------------
  // SPEED
  // ----------------------------------------------------------

  if (
    strncmp(command, "S,", 2) == 0
  )
  {
    processSpeedCommand(
      command
    );

    return;
  }


  // ----------------------------------------------------------
  // JOINT
  // ----------------------------------------------------------

  if (
    strncmp(command, "J,", 2) == 0
  )
  {
    processJointCommand(
      command
    );

    return;
  }
}


// ============================================================
// SERIAL RECEIVER
// ============================================================
//
// Fixed character buffer.
// No Arduino String.
// No response for every J command.
//
// ============================================================

void readSerial()
{
  while (
    Serial.available() > 0
  )
  {
    char c =
      Serial.read();


    // --------------------------------------------------------
    // End of command
    // --------------------------------------------------------

    if (
      c == '\n' ||
      c == '\r'
    )
    {
      if (
        serialIndex > 0
      )
      {
        serialBuffer[
          serialIndex
        ] = '\0';


        processCommand(
          serialBuffer
        );


        serialIndex = 0;
      }
    }


    // --------------------------------------------------------
    // Store character
    // --------------------------------------------------------

    else
    {
      if (
        serialIndex <
        SERIAL_BUFFER_SIZE - 1
      )
      {
        serialBuffer[
          serialIndex++
        ] = c;
      }
      else
      {
        serialIndex = 0;

        Serial.println(
          "ERROR,BUFFER_OVERFLOW"
        );
      }
    }
  }
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  // ----------------------------------------------------------
  // SERIAL
  // ----------------------------------------------------------

  Serial.begin(
    SERIAL_BAUD
  );


  // ----------------------------------------------------------
  // I2C
  // ----------------------------------------------------------

  Wire.begin();

  Wire.setClock(
    400000L
  );


  // ----------------------------------------------------------
  // PCA9685
  // ----------------------------------------------------------

  pwm.begin();

  pwm.setPWMFreq(
    SERVO_FREQ
  );

  delay(100);


  // ----------------------------------------------------------
  // INITIALIZE ALL 9 PHYSICAL SERVOS
  // ----------------------------------------------------------

  for (
    uint8_t channel = 0;
    channel < NUM_SERVOS;
    channel++
  )
  {
    currentAngle[channel] =
      0.0;

    targetAngle[channel] =
      0.0;

    currentPulse[channel] =
      CENTER_US;

    writeServoPWM(
      channel,
      0.0
    );

    delay(20);
  }


  // ----------------------------------------------------------
  // CLEAR SERIAL BUFFER
  // ----------------------------------------------------------

  while (
    Serial.available() > 0
  )
  {
    Serial.read();
  }

  serialIndex = 0;


  // ----------------------------------------------------------
  // STARTUP MESSAGE
  // ----------------------------------------------------------

  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.println(
    " 6-DOF ROBOT + GRIPPER"
  );

  Serial.println(
    " ARDUINO UNO + PCA9685"
  );

  Serial.println(
    "========================================"
  );

  Serial.println();

  Serial.println(
    "UART: 115200"
  );

  Serial.println(
    "I2C: 400 kHz"
  );

  Serial.println(
    "PCA9685: 0x40"
  );

  Serial.println(
    "Servo PWM: 50 Hz"
  );

  Serial.println(
    "PWM: 500 - 2500 us"
  );

  Serial.println(
    "ANGLE: -180 to +180"
  );

  Serial.println();

  Serial.println(
    "CH0 -> BASE"
  );

  Serial.println(
    "CH1 + CH2 -> J1"
  );

  Serial.println(
    "CH3 -> J2"
  );

  Serial.println(
    "CH4 -> J3"
  );

  Serial.println(
    "CH5 -> J4"
  );

  Serial.println(
    "CH6 -> J5"
  );

  Serial.println(
    "CH7 -> J6"
  );

  Serial.println(
    "CH8 -> GRIPPER"
  );

  Serial.println();

  Serial.println(
    "SPEED 1  = 50 deg/s"
  );

  Serial.println(
    "SPEED 10 = 500 deg/s"
  );

  Serial.println();

  Serial.println(
    "READY"
  );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  // Receive commands continuously
  readSerial();

  // Update servo positions continuously
  updateMotion();
}