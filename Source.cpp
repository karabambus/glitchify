#include <iostream>
#include <sndfile.h>
#include <opencv2/opencv.hpp>
#include <tuple>

using namespace std;
using namespace cv;


// This function calculates the sign of a value
template <typename T> int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

class LowPassFilter
{
public:
    // Constructor
    LowPassFilter(float cutoff)
    {
        m_cutoff = cutoff;
        m_y1 = 0;
    }

    // Processes a sample of the input signal and returns the filtered sample
    float process(float x0)
    {
        // Calculate the filtered sample using the first-order low-pass filter formula
        float y0 = x0 * (1 - m_cutoff) + m_y1 * m_cutoff;

        // Update the state variables
        m_y1 = y0;

        return y0;
    }

private:
    // State variables
    float m_cutoff;
    float m_y1;
};


// This function applies the bass boost effect to the input data and returns the processed data array
tuple <short*, short*, short*> applyBassBoost(const short* Blues, const short* Greens, const short* Reds, int numSamples, float boost, float cutoff)
{
    // Create the output sound data array
    short* outputBlues = new short[numSamples];
    short* outputGreens = new short[numSamples];
    short* outputReds = new short[numSamples];

    // Create the low-pass filter
    LowPassFilter filter(cutoff);

    // Iterate over the input data and apply the bass boost effect
    for (int i = 0; i < numSamples; i++)
    {
        // Filter the input signal using the low-pass filter
        short filteredBlue = filter.process(Blues[i]);
        short filteredGreen = filter.process(Greens[i]);
        short filteredRed = filter.process(Reds[i]);

        // Apply the bass boost effect by boosting the low frequencies in the filtered signal
        // using the boost parameter to control the amount of boost applied
        outputBlues[i] = filteredBlue + boost * filteredBlue;
        outputGreens[i] = filteredGreen + boost * filteredGreen;
        outputReds[i] = filteredRed + boost * filteredRed;
    }

    return make_tuple(outputBlues, outputGreens, outputReds);
}



// This function applies the phaser effect to the input data and returns the processed data array
tuple <short*, short*, short*> applyPhaser(const short* Blues, const short* Greens, const short* Reds, int numSamples, int delay, float feedback, float wetDryMix, float lowPassCutoff, float oscillatorRate, float frequencyRange, float phaseShiftDepth, int waveformType)
{
    // Create the output sound data array
    short* outputBlues = new short[numSamples];
    short* outputGreens = new short[numSamples];
    short* outputReds = new short[numSamples];

    // Initialize the phase shift oscillator and feedback accumulator
    float lowPass = 0.0f;
    float oscillator = 0.0f;
    short feedbackBlue = 0;
    short feedbackGreen = 0;
    short feedbackRed = 0;

    // Iterate over the input sound data and apply the phaser effect
    for (int i = 0; i < numSamples; i++)
    {
        // Calculate the sample index to use for the phaser effect
        int phaseShiftIndex = i - delay;

        // Check if the phase shift index is within the valid range of the input sound data
        if (phaseShiftIndex < 0)
        {
            // If the phase shift index is outside the range, set it to the first sample in the input data
            phaseShiftIndex = 0;
        }
        else if (phaseShiftIndex >= numSamples)
        {
            // If the phase shift index is outside the range, set it to the last sample in the input data
            phaseShiftIndex = numSamples - 1;
        }

        // Apply the phaser effect by adding the phase shifted sample value and feedback to the current sample value
        // and mixing the wet and dry signals based on the wetDryMix parameter
        outputBlues[i] = (Blues[i] * (1.0f - wetDryMix)) + (Blues[phaseShiftIndex] + feedbackBlue) * wetDryMix;
        outputGreens[i] = (Greens[i] * (1.0f - wetDryMix)) + (Greens[phaseShiftIndex] + feedbackGreen) * wetDryMix;
        outputReds[i] = (Reds[i] * (1.0f - wetDryMix)) + (Reds[phaseShiftIndex] + feedbackRed) * wetDryMix;

        // Apply the low-pass filter to the output signal
        lowPass = lowPassCutoff * lowPass + (1.0f - lowPassCutoff) * outputBlues[i];
        outputBlues[i] = lowPass;
        lowPass = lowPassCutoff * lowPass + (1.0f - lowPassCutoff) * outputGreens[i];
        outputGreens[i] = lowPass;
        lowPass = lowPassCutoff * lowPass + (1.0f - lowPassCutoff) * outputReds[i];
        outputReds[i] = lowPass;

        // Update the phase shift oscillator and feedback accumulator
        oscillator += oscillatorRate;
        switch (waveformType)
        {
        case 0: // Sine wave
            feedbackBlue = feedback * outputBlues[i] * sin(oscillator) * frequencyRange * phaseShiftDepth;
            feedbackGreen = feedback * outputGreens[i] * sin(oscillator) * frequencyRange * phaseShiftDepth;
            feedbackRed = feedback * outputReds[i] * sin(oscillator) * frequencyRange * phaseShiftDepth;
            break;
        case 1: // Square wave
            feedbackBlue = feedback * outputBlues[i] * sign(sin(oscillator)) * frequencyRange * phaseShiftDepth;
            feedbackGreen = feedback * outputGreens[i] * sign(sin(oscillator)) * frequencyRange * phaseShiftDepth;
            feedbackRed = feedback * outputReds[i] * sign(sin(oscillator)) * frequencyRange * phaseShiftDepth;
            break;
        case 2: // Sawtooth wave
            feedbackBlue = feedback * outputBlues[i] * (oscillator - floor(oscillator + 0.5)) * frequencyRange * phaseShiftDepth;
            feedbackGreen = feedback * outputGreens[i] * (oscillator - floor(oscillator + 0.5)) * frequencyRange * phaseShiftDepth;
            feedbackRed = feedback * outputReds[i] * (oscillator - floor(oscillator + 0.5)) * frequencyRange * phaseShiftDepth;
            break;
        default: // Sine wave (default)
            feedbackBlue = feedback * outputBlues[i] * sin(oscillator) * frequencyRange * phaseShiftDepth;
            feedbackGreen = feedback * outputGreens[i] * sin(oscillator) * frequencyRange * phaseShiftDepth;
            feedbackRed = feedback * outputReds[i] * sin(oscillator) * frequencyRange * phaseShiftDepth;
            break;
        }
        
    }
    return make_tuple(outputBlues, outputGreens, outputReds);
}



// This function applies the echo effect to the input data and returns the processed data array
tuple <short*, short*, short*> applyEcho(const short* Blues, const short* Greens, const short* Reds, int numSamples, int delay, float feedback, float dryWetMix)
{
    // Create the output sound data array
    short* outputBlues = new short[numSamples];
    short* outputGreens = new short[numSamples];
    short* outputReds = new short[numSamples];

    // Initiliaze feedback accumulator
    short feedbackBlue = 0;
    short feedbackGreen = 0;
    short feedbackRed = 0;

    // Iterate over the input sound data and apply the echo effect
    for (int i = 0; i < numSamples; i++)
    {
        // Calculate the sample index to use for the echo effect
        int delayIndex = i - delay;

        // Check if the delay index is within the valid range of the input sound data
        if (delayIndex < 0)
        {
            // If the delay index is outside the range, set it to the first sample in the input data
            delayIndex = 0;
        }
        else if (delayIndex >= numSamples)
        {
            // If the delay index is outside the range, set it to the last sample in the input data
            delayIndex = numSamples - 1;
        }

        // Apply the echo effect by adding the delayed sample value and feedback to the current sample value
        // and mixing the wet and dry signals based on the dryWetMix parameter
        outputBlues[i] = (Blues[i] * (1.0f - dryWetMix)) + (Blues[delayIndex] + feedbackBlue) * dryWetMix;
        outputGreens[i] = (Greens[i] * (1.0f - dryWetMix)) + (Greens[delayIndex] + feedbackGreen) * dryWetMix;
        outputReds[i] = (Reds[i] * (1.0f - dryWetMix)) + (Reds[delayIndex] + feedbackRed) * dryWetMix;

        // Update the feedback accumulator
        feedbackBlue = feedback * outputBlues[i];
        feedbackGreen = feedback * outputGreens[i];
        feedbackRed = feedback * outputReds[i];
    }

    return make_tuple(outputBlues, outputGreens, outputReds);
}

// This function reads an image file, using the OpenCV imread() function, and returns the image data
Mat readImage(const string& filename)
{
    Mat image = imread(filename);
    if (image.empty())
    {
        cerr << "Failed to load image: " << filename << endl;
        exit(1);
    }

    return image;
}

// This function seperates blue, green, red pixels from the input image, and returns 3 data array
tuple <short*, short*, short*> generateImageSamples(const Mat& image, int& numSamples)
{
    // Calculate the number of samples needed to represent each pixel in the input image
    numSamples = image.cols * image.rows;

    // Create the output data array
    short* Blues = new short[numSamples];
    short* greens = new short[numSamples];
    short* reds = new short[numSamples];

    // Iterate over the input image pixels and generate output data
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            // Get the pixel value at the current position
            Vec3b pixel = image.at<Vec3b>(y, x);

            // Map the pixel value to a sample value
            short Blue = pixel[0];
            short green = pixel[1];
            short red = pixel[2];

            // Write the samples to the output data array
            for (int i = 0; i < red; i++)
            {
                Blues[y * image.cols + x + i] = Blue;
            }
            for (int i = 0; i < Blue; i++)
            {
                greens[y * image.cols + x + i] = green;
            }
            for (int i = 0; i < green; i++)
            {
                reds[y * image.cols + x + i] = red;
            }
        }
    }

      

    return make_tuple(Blues, greens, reds);
}

// This function applies the delay effect to the input data and returns the processed data array
tuple <short*, short*, short*> applyDelay(const short* Blues, const short* Greens, const short* Reds, int numSamples, int delay, float feedback)
{
    // Create the output sound data array
    short* outputBlues = new short[numSamples];
    short* outputGreens = new short[numSamples];
    short* outputReds = new short[numSamples];

    //Initiliaze feedback accumulator
    short feedbackBlue = 0;
    short feedbackGreen = 0;
    short feedbackRed = 0;

    // Iterate over the input sound data and apply the delay effect
    for (int i = 0; i < numSamples; i++)
    {
        // Calculate the sample index to use for the delay effect
        int delayIndex = i - delay;

        // Check if the delay index is within the valid range of the input sound data
        if (delayIndex < 0)
        {
            // If the delay index is outside the range, set it to the first sample in the input data
            delayIndex = 0;
        }
        else if (delayIndex >= numSamples)
        {
            // If the delay index is outside the range, set it to the last sample in the input data
            delayIndex = numSamples - 1;
        }

        // Apply the delay effect by adding the delayed sample value and feedback to the current sample value 
        outputBlues[i] = Blues[i] + Blues[delayIndex] + feedbackBlue;
        outputGreens[i] = Greens[i] + Greens[delayIndex] + feedbackGreen;
        outputReds[i] = Reds[i] + Reds[delayIndex] + feedbackRed;

        // Update the feedback accumulator
        feedbackBlue = feedback * outputBlues[i];
        feedbackGreen = feedback * outputGreens[i];
        feedbackRed = feedback * outputReds[i];
    }

    return make_tuple(outputBlues, outputGreens, outputReds);
}


// This function generates the output image from the modified data split into RGB
Vec3b* generateImagePixels(const short* blues, const short* greens, const short* reds, int numSamples, int imageWidth, int imageHeight)
{
    // Create the output image data array
    Vec3b* pixels = new Vec3b[imageWidth * imageHeight];

    // Iterate over the input data and generate output image pixels
    int pixelIndex = 0;
    for (int i = 0; i < numSamples; i++)
    {
        // Get the sound sample value at the current position
        short blue = blues[i];
        short green = greens[i];
        short red = reds[i];

        // Map the sample value to an image pixel value
        Vec3b pixel (green, blue, red);

        // Write the image pixel to the output data array
        pixels[pixelIndex++] = pixel;
    }

    return pixels;
}

// This function creates an image from the input image data and saves it to the specified file
void writeImageFile(const string& filename, const Vec3b* pixels, int imageWidth, int imageHeight)
{
    // Create the output image from the input image data
    Mat image(imageHeight, imageWidth, CV_8UC3, (void*)pixels);

    // Save the output image to the specified file
    if (!imwrite(filename, image))
    {
        cerr << "Failed to save image: " << filename << endl;
        exit(1);
    }
}

// This is the main entry point of the program, which orchestrates the other functions to perform effects opeartions
// To do - let user input values and filename instead of hard coding it
int main()
{
 
    string inputImageFilename = "test.jpg";
    string outputImageFilename = "testOut.jpg";
    string outputSoundFilename = "testsound.wav";

    // Read the input image file and get the image data
    Mat image = readImage(inputImageFilename);
    

   //intilize values for splitting the image data in to 3 datta arrays BLUE GREEN RED
    int numSamples = image.cols * image.rows;
    short* blues;
    short* greens;
    short* reds;
    
    // Split pixel data to BLUE GREEN RED channels, and add values of pixels respectivly 
    tie (blues, greens, reds) = generateImageSamples(image, numSamples);

    // choose which effect to turn on
    bool echoEffect = 0;
    bool delayEffect = 0;
    bool phaserEffect = 0;

    // Initialize values for  effects
    int delay = 500; // large numbers
    int waveformType = 3; // 0 = sin, 1 = saw, 3 = square
    float feedback = 0;  //from 0 to 1
    float dryWetMix = 0.5; //from 0 to 1
    float lowPassCutoff = 0.5; //from 0 to 1
    float oscillatorRate = 0.7;
    float frequencyRange = 1;
    float phaseShiftDepth = 0.9;
    float boost = 0.9;
    short* outputBlues = 0;
    short* outputGreens = 0;
    short* outputReds = 0;

    tie(outputBlues, outputGreens, outputReds) = applyBassBoost(blues, greens, reds, numSamples, boost, lowPassCutoff);

    if (phaserEffect == 1)
    tie(outputBlues, outputGreens, outputReds) = applyPhaser(blues, greens, reds, numSamples, delay, feedback, dryWetMix, lowPassCutoff, oscillatorRate, frequencyRange, phaseShiftDepth, waveformType);

    //apply echo effect to data
    if (echoEffect == 1)
    tie(outputBlues, outputGreens, outputReds) = applyEcho(blues, greens, reds, numSamples, delay, feedback, dryWetMix);
    
    // Apply the delay effect to data
    if (delayEffect == 1)
    tie(outputBlues, outputGreens, outputReds) = applyDelay(blues, greens, reds, numSamples, delay, feedback);

    // Generate the output image data from the audio data
    Vec3b* pixels = generateImagePixels(outputBlues, outputGreens, outputReds, numSamples, image.cols, image.rows);

    // Write the output image data to the output image file
    writeImageFile(outputImageFilename, pixels, image.cols, image.rows);

    return 0;
}

