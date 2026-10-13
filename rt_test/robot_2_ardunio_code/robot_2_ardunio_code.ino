#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// ROBOT 2
// Arduino UNO + 16-Channel I2C Servo Driver
//
// ROS2 UART NODE — UNCHANGED
//
// RECEIVED COMMANDS:
//
// J,channel,angle
// S,speed
// HOME
// STOP
//
// CHANNELS:
//
// CH0 -> BASE
// CH1 -> J1
// CH2 -> J2
// CH3 -> J3
// CH4 -> J4
// CH5 -> GRIPPER
// ============================================================


// ============================================================
// PCA9685
// ============================================================

#define PCA9685_ADDR 0x40

Adafruit_PWMServoDriver pwm(PCA9685_ADDR);


// ============================================================
// SERVO CHANNELS
// ============================================================

#define CH_BASE       0
#define CH_J1         1
#define CH_J2         2
#define CH_J3         3
#define CH_J4         4
#define CH_GRIPPER    5

#define NUM_SERVOS    6


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

// Motion update every 5 ms

#define MOTION_UPDATE_MS 5

unsigned long lastMotionUpdate = 0;


// ============================================================
// SPEED SETTINGS
// ============================================================
//
// ROS2 sends:
//
// S,1
// S,2
// ...
// S,10
//
// Actual angular velocity:
//
// 1  = 50 deg/s
// 2  = 100 deg/s
// 3  = 150 deg/s
// 4  = 200 deg/s
// 5  = 250 deg/s
// 6  = 300 deg/s
// 7  = 350 deg/s
// 8  = 400 deg/s
// 9  = 450 deg/s
// 10 = 500 deg/s
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
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0
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
  CENTER_US
};


// ============================================================
// MOTION ENABLE
// ============================================================

bool motionEnabled = true;


// ============================================================
// ANGLE → PWM MICROSECONDS
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
// SET TARGET ANGLE
// ============================================================
//
// IMPORTANT:
//
// This function ONLY changes the target.
//
// It does NOT print anything to Serial.
//
// This is important because ROS2 continuously sends
// J commands and the Arduino does not need to send
// a response for every J command.
//
// ============================================================

void setServoTarget(
  uint8_t channel,
  float angle
)
{
  if (channel >= NUM_SERVOS)
    return;


  // General servo range

  angle = constrain(
    angle,
    -180.0,
    180.0
  );



  targetAngle[channel] =
    angle;


  // New command enables motion

  motionEnabled = true;
}


// ============================================================
// UPDATE MOTION
// ============================================================
//
// Non-blocking.
//
// Every 5 ms the current angle moves toward the target.
//
// New ROS2 commands can arrive at any time.
//
// No command queue is created.
//
// The newest target always replaces the previous target.
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
    // Safety limits
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
  // Freeze at current commanded position

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
//
// This is only used when manually sending P.
// It is NOT called for normal J commands.
//
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


  // Only one short response when speed changes

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
// J,0,30
// J,1,-20
// J,2,45
// J,3,10
// J,4,-15
// J,5,-55
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


  // Separate channel and angle

  *secondComma = '\0';


  int channel =
    atoi(
      firstComma + 1
    );


  float angle =
    atof(
      secondComma + 1
    );


  if (
    channel < 0 ||
    channel >= NUM_SERVOS
  )
  {
    return;
  }


  // Set target

  setServoTarget(
    channel,
    angle
  );
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
//
// No Arduino String.
//
// This prevents dynamic memory allocation/fragmentation.
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
        // Reset buffer if command is too long

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

  // Fast I2C

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
  // INITIALIZE ALL SERVOS
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
    " ROBOT 2 - ARDUINO UNO"
  );

  Serial.println(
    " 16-CHANNEL SERVO DRIVER"
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
    "CENTER: 1500 us"
  );

  Serial.println();

  Serial.println(
    "CH0 -> BASE"
  );

  Serial.println(
    "CH1 -> J1"
  );

  Serial.println(
    "CH2 -> J2"
  );

  Serial.println(
    "CH3 -> J3"
  );

  Serial.println(
    "CH4 -> J4"
  );

  Serial.println(
    "CH5 -> GRIPPER"
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
  // Receive ROS2 commands continuously

  readSerial();


  // Update servo positions continuously

  updateMotion();
}