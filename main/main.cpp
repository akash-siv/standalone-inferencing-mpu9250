/* Edge Impulse Espressif ESP32 Standalone Inference ESP IDF Example
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Include ----------------------------------------------------------------- */
#include <stdio.h>

#include "ei_run_classifier.h"

#include "driver/gpio.h"
#include "sdkconfig.h"

// for mpu9250
#include <stdint.h>
#include <stdlib.h>
#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "I2Cbus.hpp"
#include "MPU.hpp"
#include "mpu/math.hpp"
#include "mpu/types.hpp"

#define LED_PIN GPIO_NUM_2

// for mpu9250
static const char* ML = "ML";
static const char* IMU = "ML";
static constexpr gpio_num_t SDA = GPIO_NUM_21;
static constexpr gpio_num_t SCL = GPIO_NUM_22;
static constexpr uint32_t CLOCK_SPEED = 400000;  // range from 100 KHz ~ 400Hz
// end mpu9250


static const float features[] = {
    // copy raw features here (for example from the 'Live classification' page)
    -6.8142, 0.8664, 1.6813, -7.4157, 1.4445, 2.2130, -4.7344, 1.9637, 1.9334, -3.7764, 2.2864, 1.9268, -2.9279, 1.4534, 2.9621, -1.1214, 0.9097, 1.9990, 1.1096, 0.5496, 1.5169, 3.1473, 0.6517, 1.4927, 5.3965, 1.2116, 1.7643, 7.6396, 1.8844, 3.3438, 11.3581, 3.4046, 2.9771, 14.0510, 3.5889, 2.9877, 14.5196, 2.2113, 2.7881, 13.7868, 0.9325, 2.0222, 14.2978, 1.4598, 0.9477, 15.3975, 2.9354, 1.8981, 16.5699, 3.8591, 3.4187, 17.0762, 3.9515, 4.6817, 17.1629, 4.0186, 5.1835, 17.1570, 4.0222, 4.8213, 17.0805, 4.0067, 3.9559, 16.7896, 3.3999, 2.8797, 16.2628, 3.2018, 2.2466, 15.5132, 3.8019, 2.1880, 14.4155, 4.2643, 1.8879, 13.3237, 4.0540, 2.4672, 11.9856, 3.5973, 2.7202, 10.3786, 3.4319, 2.5688, 8.5053, 3.4985, 2.2607, 6.5239, 3.6304, 2.0658, 4.4948, 3.3355, 2.0215, 2.7674, 3.3207, 1.9715, 1.2969, 3.6302, 2.2723, -0.2468, 3.8486, 2.7761, -1.8201, 3.7643, 2.6183, -4.1894, 3.2432, 2.4075, -6.0904, 3.0711, 2.1480, -7.5694, 3.3559, 2.0473, -8.5238, 3.7625, 1.9919, -9.0578, 3.8673, 1.9232, -9.2895, 3.0184, 1.7867, -10.8429, 1.7583, 1.5910, -13.0116, 1.0173, 1.3409, -15.1116, 0.7930, 1.1667, -17.4247, 0.0923, 0.6937, -19.0956, -0.2249, 1.1805, -19.4660, -0.3979, 0.9585, -19.2506, -0.7707, 0.4865, -19.4566, -1.3550, 0.1209, -19.8423, -1.4928, -0.5113, -19.4017, -1.2594, -0.7522, -18.1755, -1.1508, -0.5946, -16.7448, -1.0022, -0.2920, -15.4634, -0.4783, -0.2094, -14.5930, -0.1630, 0.1939, -13.3936, 0.5994, 0.3674, -12.5918, 0.9291, 1.1047, -11.2881, 1.3732, 1.6481, -9.1767, 2.0721, 1.8532, -7.4715, 2.2394, 2.7165, -5.6988, 2.2874, 2.6248, -4.2701, 2.0013, 2.3728, -2.6302, 1.9264, 2.1773, -0.8752, 2.0125, 2.5546, 1.0521, 2.0878, 3.4284, 3.3089, 1.5127, 3.7865, 6.6519, 1.3658, 3.4052, 9.6756, 1.3612, 2.8678, 11.5641, 1.4012, 2.5182, 13.1395, 1.6840, 1.9973, 14.9944, 2.2647, 1.8324, 16.1733, 2.2239, 2.1697, 16.5999, 2.4339, 2.8085, 16.7507, 3.3095, 3.2709, 16.7336, 4.0513, 3.6436, 16.4432, 3.9701, 3.5848, 16.1510, 4.0335, 3.6542, 16.2164, 4.1921, 3.6578, 16.3703, 3.9928, 3.4573, 15.9537, 3.8586, 3.0711, 15.2814, 3.5679, 2.5164, 14.5031, 3.7516, 1.8643, 13.4785, 3.7712, 1.7507, 12.2270, 3.4886, 2.2062, 11.2821, 3.6480, 2.2473, 10.1597, 4.0031, 2.1783, 8.6548, 4.2187, 1.8533, 6.6399, 3.8987, 1.7386, 4.6232, 3.3737, 1.6926, 2.9005, 3.1964, 1.5842, 1.3152, 3.0632, 1.4348, -0.6473, 3.0829, 1.9569, -2.4302, 3.3734, 2.5393, -3.6705, 3.8740, 2.7195, -5.4703, 3.6907, 3.3671, -6.0191, 3.5803, 3.1104, -7.4641, 2.5092, 2.9302, -9.6437, 1.3054, 2.4420, -11.7159, 0.8752, 1.9784, -12.7759, 1.3544, 2.0011, -13.1641, 0.9449, 2.3643, -13.6052, -0.2859, 1.8856, -15.2225, -2.1116, 0.8173, -17.7846, -3.6724, -0.1103, -19.9044, -3.8263, -1.3056, -20.4459, -3.9394, -1.2218, -20.0538, -3.5280, -1.4674, -19.5691, -3.5087, -1.7898, -19.2803, -4.0276, -1.9392, -19.1735, -4.3842, -1.6546, -18.0779, -4.0995, -1.8833, -16.7603, -3.5558, -1.5015, -15.4101, -2.9203, -1.0540, -14.3050, -2.2116, -0.6783, -13.2805, -1.4628, -0.2138, -12.6262, -1.2342, 0.8711, -11.0761, -0.8067, 1.0274, -9.3690, 0.0970, 1.1878, -8.0629, 1.2215, 1.8778, -6.3701, 2.1448, 1.9201, -5.3236, 2.6541, 2.4959, -3.8488, 3.0705, 3.3287, -1.7440, 3.1527, 4.1675, 0.5587, 2.4654, 4.3809, 2.0965, 1.2550, 4.2530
};

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}


extern "C" int app_main()
{   
    // for mpu9250
    ESP_LOGI(IMU, "$ MPU Driver Initializing: MPU-I2C_9250\n");

    // Initialize I2C on port 0 using I2Cbus interface
    i2c0.begin(SDA, SCL, CLOCK_SPEED);

    MPU_t MPU;  // create a default MPU object
    MPU.setBus(i2c0);  // set bus port, not really needed since default is i2c0
    MPU.setAddr(mpud::MPU_I2CADDRESS_AD0_LOW);  // set address, default is AD0_LOW

    // (this also check if the connected MPU supports the implementation of chip selected in the component menu)
    while (esp_err_t err = MPU.testConnection()) {
        ESP_LOGE(IMU, "Failed to connect to the MPU, error=%#X", err);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    ESP_LOGI(IMU, "MPU connection successful!");

    // Initialize
    ESP_ERROR_CHECK(MPU.initialize());  // initialize the chip and set initial configurations
    // Setup with your configurations
    ESP_ERROR_CHECK(MPU.setSampleRate(50));  // set sample rate to 50 Hz

    // Reading sensor data
    mpud::raw_axes_t accelRaw;   // x, y, z axes as int16
    mpud::float_axes_t accelG;   // accel axes in (g) gravity format

    // end mpu9250

    gpio_pad_select_gpio(LED_PIN);
    gpio_reset_pin(LED_PIN);  

    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT); 

    ei_sleep(100);

    ei_impulse_result_t result = { nullptr };

    ei_printf("Edge Impulse standalone inferencing (Espressif ESP32)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    {
        ei_printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    }

    while (true)
    {
        // Read mpu9250
        MPU.acceleration(&accelRaw);  // fetch raw data from the registers
        // Convert
        accelG = mpud::accelGravity(accelRaw, mpud::ACCEL_FS_4G);
        // Debug
        printf("%.4f,%.4f,%.4f\n", accelG.x*10, accelG.y*10, accelG.z*10);
        vTaskDelay(10/ portTICK_PERIOD_MS);
        // End Mpu9250

        // blink LED
        gpio_set_level(LED_PIN, 1);

        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);

        ei_printf("run_classifier returned: %d\n", res);

        if (res != 0)
        return 1;

        ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);

        // print the predictions
        ei_printf("[");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
        {
        ei_printf("%.5f", result.classification[ix].value);
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf(", ");
    #else
        if (ix != EI_CLASSIFIER_LABEL_COUNT - 1)
        {
            ei_printf(", ");
        }
    #endif
        }
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("%.3f", result.anomaly);
    #endif
        printf("]\n");

        gpio_set_level(LED_PIN, 0);
        ei_sleep(1000);
    }
}

// 
// * dont use printf use LOG instead.

/* 
*Todo:
*dont use printf use LOG instead.
*
*/