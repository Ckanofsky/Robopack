#include <Servo.h>

Servo LeftMotorPWM;
Servo RightMotorPWM;

#define LeftMotorLED D9
#define RightMotorLED D1

String cmd = "";
int LeftMotorPower = 110;
int RightMotorPower = 110;
bool NewCmd = false;

void setup() {
  pinMode(LeftMotorLED, OUTPUT);
  pinMode(RightMotorLED, OUTPUT);

  LeftMotorPWM.attach(D0);
  RightMotorPWM.attach(D10);

  LeftMotorPWM.write(90);
  RightMotorPWM.write(90);

  Serial.begin(9600);
}

void loop() {
  readInput();
  setPower(LeftMotorPower, RightMotorPower);
}

void readInput(){
  if (Serial.available() && Serial.find("M")) {
    NewCmd = true;
    Serial.find("R");
    RightMotorPower = Serial.parseInt();

    Serial.find("L");
    LeftMotorPower = Serial.parseInt();
  }
  if(NewCmd){
    Serial.println("R" + String(RightMotorPower) + " L" + String(LeftMotorPower));
    NewCmd = false;
  }
}

void setPower(int LP, int RP){
  LeftMotorPWM.write(LP);
  RightMotorPWM.write(RP);
  if(LP != 90){
    digitalWrite(LeftMotorLED, HIGH);
  } else {
    digitalWrite(LeftMotorLED, LOW);
  }
  if(RP != 90){
    digitalWrite(RightMotorLED, HIGH);
  } else {
    digitalWrite(RightMotorLED, LOW);
  }
}