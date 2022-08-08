/* Edge Impulse Espressif ESP32 Standalone Inference ESP IDF Example
 * Copyright (c) 2022 EdgeImpulse Inc.
 */

// edge-impulse-data-forwarder  //data forwarder command
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
static constexpr gpio_num_t SDA = GPIO_NUM_21;
static constexpr gpio_num_t SCL = GPIO_NUM_22;
static constexpr uint32_t CLOCK_SPEED = 400000; // range from 100 KHz ~ 400Hz
// end mpu9250

float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {};
// copy raw features here (for example from the 'Live classification' page)

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

extern "C" int app_main()
{
    // for mpu9250
    printf("$ MPU Driver Initializing: MPU-I2C_9250\n");

    // Initialize I2C on port 0 using I2Cbus interface
    i2c0.begin(SDA, SCL, CLOCK_SPEED);

    MPU_t MPU;                                 // create a default MPU object
    MPU.setBus(i2c0);                          // set bus port, not really needed since default is i2c0
    MPU.setAddr(mpud::MPU_I2CADDRESS_AD0_LOW); // set address, default is AD0_LOW

    // (this also check if the connected MPU supports the implementation of chip selected in the component menu)
    while (esp_err_t err = MPU.testConnection())
    {
        printf("Failed to connect to the MPU, error=%#X", err);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    printf("MPU connection successful!");

    // Initialize
    ESP_ERROR_CHECK(MPU.initialize()); // initialize the chip and set initial configurations
    // Setup with your configurations
    ESP_ERROR_CHECK(MPU.setSampleRate(50)); // set sample rate to 50 Hz

    // Reading sensor data
    mpud::raw_axes_t accelRaw; // x, y, z axes as int16
    mpud::float_axes_t accelG; // accel axes in (g) gravity format

    // end mpu9250

    gpio_pad_select_gpio(LED_PIN);
    gpio_reset_pin(LED_PIN);

    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    ei_sleep(100);

    ei_impulse_result_t result = {nullptr};

    ei_printf("Edge Impulse standalone inferencing (Espressif ESP32)\n");

    // if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    // {
    //     ei_printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
    //               EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
    //     return 1;
    // }

    while (true)
    {
        // Read mpu9250

        // Debug

        // vTaskDelay(10 / portTICK_PERIOD_MS);
        // End Mpu9250
        for (int j = 0; j < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; j += 3)
        {
            MPU.acceleration(&accelRaw); // fetch raw data from the registers
            // Convert
            accelG = mpud::accelGravity(accelRaw, mpud::ACCEL_FS_4G);
            features[j + 0] = accelG.x * 10;
            features[j + 1] = accelG.y * 10;
            features[j + 2] = accelG.z * 10;
            // printf("%.4f,%.4f,%.4f\n", accelG.x * 10, accelG.y * 10, accelG.z * 10);
            vTaskDelay(10 / portTICK_PERIOD_MS);
        }
        // printf("Size of the features %u\n", features);

        // blink LED
        gpio_set_level(LED_PIN, 1);

        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);

        ei_printf("run_classifier returned: %d\n", res); // used to denote the error

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
    }
}

// Todo : dont use printf use LOG instead.
