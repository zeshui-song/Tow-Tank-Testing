#include "HX711.h"

const int DRAG_DOUT_PIN = 4; 
const int DRAG_CLK_PIN = 5;  
const int LIFT_DOUT_PIN = 6; 
const int LIFT_CLK_PIN = 7;  

HX711 drag_scale;
HX711 lift_scale;

void setup() {
  Serial.begin(115200);
  
  drag_scale.begin(DRAG_DOUT_PIN, DRAG_CLK_PIN);
  lift_scale.begin(LIFT_DOUT_PIN, LIFT_CLK_PIN);
  
  delay(500); // Settle chips
}

void loop() {
  if (drag_scale.wait_ready_timeout(50) && lift_scale.wait_ready_timeout(50)) {
    // Print clean comma-separated raw values
    Serial.print(drag_scale.read());
    Serial.print(",");
    Serial.println(lift_scale.read());
  }
  
  delay(100); // Stable ~10Hz sampling rate
}