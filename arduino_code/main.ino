#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>
#include <PZEM004Tv30.h>
#include <Wire.h>

#define RX_PIN 2
#define TX_PIN 3

SoftwareSerial pzemSWSerial(RX_PIN, TX_PIN);
PZEM004Tv30 pzem(pzemSWSerial);

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int relay1 = 8;
const int relay2 = 9;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1000);

  pinMode(relay1, OUTPUT);
  pinMode(relay2, OUTPUT);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Waiting...");
}

void loop() {
  if (Serial.available() > 0) {
    int cnt = Serial.parseInt();  // Read count from Python

    if (cnt > 0) {
      digitalWrite(relay1, HIGH);
      digitalWrite(relay2, HIGH);
    } else {
      delay(15000);
      digitalWrite(relay1, LOW);
      digitalWrite(relay2, LOW);
    }
  }

  // Read sensor data
  float voltage = pzem.voltage();
  float current = pzem.current();
  float power = pzem.power();
  float energy = pzem.energy();

  // Check for errors
  if (isnan(voltage) || isnan(current) || isnan(power) || isnan(energy)) {
    Serial.println("Error reading PZEM data");
    return;
  }

  // Send data in a single line
  Serial.print(voltage);
  Serial.print("|");
  Serial.print(current);
  Serial.print("|");
  Serial.print(power);
  Serial.print("|");
  Serial.println(energy);

  // Display on LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("V:"); lcd.print(voltage); lcd.print("V ");
  lcd.print("C:"); lcd.print(current, 2); lcd.print("A");
  lcd.setCursor(0, 1);
  lcd.print("P:"); lcd.print(power); lcd.print("W ");
  lcd.print("E:"); lcd.print(energy, 2); lcd.print("kWh");

  delay(1000); // Send data every 1 second
}
