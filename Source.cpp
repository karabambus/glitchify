//Bugs to fix we are losing data somwhere probaly during conversion (maybe long double for output stuff and input)

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
    LowPassFilter(double cutoff)
    {
        m_cutoff = cutoff;
        m_y1 = 0;
    }

    // Processes a sample of the input signal and returns the filtered sample
    double process(double x0)
    {
        // Calculate the filtered sample using the first-order low-pass filter formula
        double y0 = x0 * (1 - m_cutoff) + m_y1 * m_cutoff;

        // Update the state variables
        m_y1 = y0;

        return y0;
    }

private:
    // State variables
    double m_cutoff;
    double m_y1;
};

tuple<double*, double*, double*> FillRemainingPixels(const double* inputBlues, const double* inputGreens, const double* inputReds, double* outputBlues, double* outputGreens, double* outputReds, int effectStart, int effectEnd, int numSamples) {

    for (int i = 0; i < effectStart; i++) {
        outputBlues[i] = inputBlues[i];
        outputGreens[i] = inputGreens[i];
        outputReds[i] = inputReds[i];
    }

    for (int i = effectEnd; i < numSamples; i++) {
        outputBlues[i] = inputBlues[i];
        outputGreens[i] = inputGreens[i];
        outputReds[i] = inputReds[i];
    }

    return make_tuple(outputBlues, outputGreens, outputReds);
}


// This function applies the change pitch effect to the input data and returns the processed data array
tuple<double*, double*, double*> changePitch(const double* Blues, const double* Greens, const double* Reds, int numSamples, double pitchShift, int effectStart, int effectEnd)
{
    // Create the output sound data array
    double* outputBlues = new double[numSamples];
    double* outputGreens = new double[numSamples];
    double* outputReds = new double[numSamples];

    // Calculate the pitch shift factor
    double pitchShiftFactor = pow(2.0, pitchShift / 12.0);

    // Iterate over the input sound data and apply the change pitch effect
    for (int i = effectStart; i < effectEnd; i++)
    {
        // Calculate the output sample index, using the pitch shift factor
        int outputIndex = (int)(pitchShiftFactor * i);

        // If the output index is within the range of the output array, set the sample value
        if (outputIndex < numSamples)
        {
            outputBlues[i] = Blues[outputIndex];
            outputGreens[i] = Greens[outputIndex];
            outputReds[i] = Reds[outputIndex];
        }
    }
    //fill remaining pixels
    tie(outputBlues, outputGreens, outputReds) = FillRemainingPixels(Blues, Greens, Reds, outputBlues, outputGreens, outputReds, effectStart, effectEnd, numSamples);

    return make_tuple(outputBlues, outputGreens, outputReds);
}



// This function applies the bass boost effect to the input data and returns the processed data array
tuple <double*, double*, double*> applyBassBoost(const double* Blues, const double* Greens, const double* Reds, int numSamples, double boost, double cutoff, int effectStart, int effectEnd)
{
    // Create the output sound data array
    double* outputBlues = new double[numSamples];
    double* outputGreens = new double[numSamples];
    double* outputReds = new double[numSamples];

    // Create the low-pass filter
    LowPassFilter filter(cutoff);

    // Iterate over the input data and apply the bass boost effect
    for (int i = effectStart; i < effectEnd; i++)
    {
        // Filter the input signal using the low-pass filter
        double filteredBlue = filter.process(Blues[i]);
        double filteredGreen = filter.process(Greens[i]);
        double filteredRed = filter.process(Reds[i]);

        // Apply the bass boost effect by boosting the low frequencies in the filtered signal
        // using the boost parameter to control the amount of boost applied
        outputBlues[i] = filteredBlue + boost * filteredBlue;
        outputGreens[i] = filteredGreen + boost * filteredGreen;
        outputReds[i] = filteredRed + boost * filteredRed;
    }

    tie(outputBlues, outputGreens, outputReds) = FillRemainingPixels(Blues, Greens, Reds, outputBlues, outputGreens, outputReds, effectStart, effectEnd, numSamples);

    return make_tuple(outputBlues, outputGreens, outputReds);
}



// This function applies the phaser effect to the input data and returns the processed data array
tuple <double*, double*, double*> applyPhaser(const double* Blues, const double* Greens, const double* Reds, int numSamples, int delay, double feedback, double wetDryMix, double lowPassCutoff, double oscillatorRate, double frequencyRange, double phaseShiftDepth, int waveformType, int effectStart, int effectEnd)
{
    // Create the output sound data array
    double* outputBlues = new double[numSamples];
    double* outputGreens = new double[numSamples];
    double* outputReds = new double[numSamples];

    // Initialize the phase shift oscillator and feedback accumulator
    double lowPass = 0.0f;
    double oscillator = 0.0f;
    double feedbackBlue = 0;
    double feedbackGreen = 0;
    double feedbackRed = 0;

    // Iterate over the input sound data and apply the phaser effect
    for (int i = effectStart; i < effectEnd; i++)
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
    //fill remaining pixels
    tie(outputBlues, outputGreens, outputReds) = FillRemainingPixels(Blues, Greens, Reds, outputBlues, outputGreens, outputReds, effectStart, effectEnd, numSamples);

    return make_tuple(outputBlues, outputGreens, outputReds);
}



// This function applies the echo effect to the input data and returns the processed data array
tuple <double*, double*, double*> applyEcho(const double* Blues, const double* Greens, const double* Reds, int numSamplesEffect, double feedback, double dryWetMix, double decay, double delayTime, int effectStart, int effectEnd, int numSamples)
{
    // Convert the delay time to the number of samples to delay by
    int sampleRate = 44100;
    double delayIncrement = 1.0f / sampleRate;
    int delaySamples = (int)(delayTime / delayIncrement);

    // Create the output data array
    double* outputBlues = new double[numSamples];
    double* outputGreens = new double[numSamples];
    double* outputReds = new double[numSamples];

    // Initiliaze feedback accumulator
    double feedbackBlue = 0;
    double feedbackGreen = 0;
    double feedbackRed = 0;

    // Iterate over the input data and apply the echo effect
    for (int i = effectStart; i < effectEnd; i++)
    {
        // Calculate the sample index to use for the echo effect
        int delayIndex = i - delaySamples;

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
        outputBlues[i] = (Blues[i] * (1.0f - dryWetMix)) + (Blues[delayIndex] * decay + feedbackBlue) * dryWetMix;
        outputGreens[i] = (Greens[i] * (1.0f - dryWetMix)) + (Greens[delayIndex] * decay + feedbackGreen) * dryWetMix;
        outputReds[i] = (Reds[i] * (1.0f - dryWetMix)) + (Reds[delayIndex] * decay + feedbackRed) * dryWetMix;

        // Update the feedback accumulator
        feedbackBlue = feedback * outputBlues[i];
        feedbackGreen = feedback * outputGreens[i];
        feedbackRed = feedback * outputReds[i];
    }

    tie(outputBlues, outputGreens, outputReds) = FillRemainingPixels(Blues, Greens, Reds, outputBlues, outputGreens, outputReds, effectStart, effectEnd, numSamples);

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
tuple <double*, double*, double*> generateDataArrays(const Mat& image, int& numSamples)
{
    // Calculate the number of samples needed to represent each pixel in the input image
    numSamples = image.cols * image.rows;

    // Create the output data array
    double* blues = new double[numSamples];
    double* greens = new double[numSamples];
    double* reds = new double[numSamples];



    // Iterate over the input image pixels and generate output data
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            // Get the pixel value at the current position
            Vec3b pixel = image.at<Vec3b>(y, x);

            // Map the pixel value to a sample value
            double Blue = pixel[0];
            double green = pixel[1];
            double red = pixel[2];

            // Write the samples to the output data array
            for (int i = 0; i < red; i++)
            {
                blues[y * (image.cols) + x + i] = Blue;
            }
            for (int i = 0; i < Blue; i++)
            {
                greens[y * (image.cols) + x + i] = green;
            }
            for (int i = 0; i < green; i++)
            {
                reds[y * (image.cols) + x + i] = red;
            }
        }
    }



    return make_tuple(blues, greens, reds);
}


// potreban je brzi algoritam kao fft, prekompleksno za meneda implentam
//// this function applies the echo effect to the input data using a convolution kernel calculated using an exponential decay function, and returns the processed data array
//tuple <double*, double*, double*> applyecho2(const double* blues, const double* greens, const double* reds, int numsamples, int delay, double decayfactor, double drywetmix)
//{
//    // create the output sound data array
//    double* outputblues = new double[numsamples];
//    double* outputgreens = new double[numsamples];
//    double* outputreds = new double[numsamples];
//
//    // calculate the length of the convolution kernel
//    int kernellength = delay + 1;
//
//    // create the convolution kernel using an exponential decay function
//    double* kernel = new double[kernellength];
//    kernel[0] = 1.0f; // initial impulse
//    for (int i = 1; i < kernellength; i++)
//    {
//        kernel[i] = kernel[i - 1] * decayfactor; // exponential decay
//    }
//
//    // iterate over the input sound data and apply the echo effect
//    for (int i = 0; i < numsamples; i++)
//    {
//        // create temporary arrays to store the convolution output for each channel
//        double* convolvedblue = new double[numsamples];
//        double* convolvedgreen = new double[numsamples];
//        double* convolvedred = new double[numsamples];
//
//        // apply the convolution operation to each channel using the kernel from the input
//        for (int j = 0; j < kernellength; j++)
//        {
//            int delayindex = delay - i;
//            // check if the delay index is within the valid range of the input sound data
//            if (delayindex < 0)
//            {
//                // if the delay index is outside the range, set it to the first sample in the input data
//                delayindex = 0;
//            }
//            else if (delayindex >= numsamples)
//            {
//                // if the delay index is outside the range, set it to the last sample in the input data
//                delayindex = numsamples - 1;
//            }
//        if (delayindex >= 0 && delayindex < kernellength)
//            {
//                convolvedblue[i] += blues[delayindex] * kernel[j];
//                convolvedgreen[i] += greens[delayindex] * kernel[j];
//                convolvedred[i] += reds[delayindex] * kernel[j];
//            }
//        }
//        // mix the wet and dry signals based on the drywetmix parameter
//        outputblues[i] = (blues[i] * (1.0f - drywetmix)) + convolvedblue[i] * drywetmix;
//        outputgreens[i] = (greens[i] * (1.0f - drywetmix)) + convolvedgreen[i] * drywetmix;
//        outputreds[i] = (reds[i] * (1.0f - drywetmix)) + convolvedred[i] * drywetmix;
//        delete[] convolvedblue;
//        delete[] convolvedgreen;
//        delete[] convolvedred;
//    }
//    delete[] kernel;
//    
//    return make_tuple(outputblues, outputgreens, outputreds);
//}



// This function generates the output image from the modified data split into RGB
Vec3b* generateImagePixels(const double* outputBlues, const double* outputGreens, const double* outputReds, int numSamples, int imageWidth, int imageHeight)
{
    // Create the output image data array
    Vec3b* pixels = new Vec3b[imageWidth * imageHeight];

    // Iterate over the input data and generate output image pixels
    int pixelIndex = 0;
    for (int i = 0; i < numSamples; i++)
    {
        // Get the color sample value at the current position
        double blue = outputBlues[i];
        double green = outputGreens[i];
        double red = outputReds[i];

        // Map the sample value to an image pixel value
        Vec3b pixel(green, blue, red);

        //Vec3b pixel(blue, green, red);

        // Write the image pixel to the output data array
        pixels[pixelIndex++] = pixel;
    }

    return pixels;
}

// This function creates an image from the input image data and saves it to the specified file
void writeImageFile(const string& filename, const Vec3b* pixels, int imageWidth, int imageHeight, bool effectDirection)
{
    // Create the output image from the input image data
    Mat image(imageHeight, imageWidth, CV_8UC3, (void*)pixels);

    //transposes data array back to its original
    if (effectDirection)
        transpose(image, image);

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

    //change effect direction (from left to right/top to down)
    bool effectDirection = 0; // 1 or 0

    //transposes data array
    if (effectDirection)
        transpose(image, image);

    //intilize values for splitting the image data in to 3 datta arrays BLUE GREEN RED
    int numSamples = image.cols * image.rows;
    cout << numSamples;
    double* blues;
    double* greens;
    double* reds;

    // Split pixel data to BLUE GREEN RED channels, and add values of pixels respectivly 
    tie(blues, greens, reds) = generateDataArrays(image, numSamples);

    // choose which effect to turn on
    bool echoEffect = 0;
    //bool delayEffect = 0;
    bool phaserEffect = 1;
    bool bassBoostEffect = 0;
    bool pitchShiftEffect = 0;

    // Initialize values for  effects
    // TODO: Too mutch values could all be replaced with 1 or 2, probably when adding user input lines is best time to do that
    float start = 0.3; // from what point to apply effect, format as precentage ie 0.02 as 2%, 0 from beggining
    float end = 0.6; // to what point effect is applied, format as precentage ie 0.02 as 2%, 1 for end
    int delay = 100; // only used for phaser
    int waveformType = 3; // 0 = sin, 1 = saw, 3 = square 
    double feedback = 1;  //from 0 to 2
    double dryWetMix = 0.5; //from 0 to 1
    double lowPassCutoff = 0.5; //from 0 to 1
    double oscillatorRate = 0.7; // 0 to 1 (probably)
    double frequencyRange = 1; // 0 to 1 (probably)
    double phaseShiftDepth = 0.9;// 0 to 1 (probably)
    double boost = 0.4; // 0 to 1 (probably)
    double decayFactor = 0.9; // 0 to 1 (probably)
    double pitchShift = 0.2; // best values are from 0 to 0.5, keep in mind you can use 0.0x 
    double decayTime = 1.5; // best values are from 0 to 1, more then that adds feedback
    double delayTime = 0.9; // best values are from 0 to 1
    double* outputBlues = 0;
    double* outputGreens = 0;
    double* outputReds = 0;


    //calculate pixel number from values "from" and "to"
    int effectStart = numSamples * start;
    int effectEnd = numSamples * end;
    cout << endl << effectStart << "\t" << effectEnd;
    int numSamplesEffect = (effectEnd - effectStart) - 2;

    //apply bass boost effect to data
    if (bassBoostEffect == 1)
        tie(outputBlues, outputGreens, outputReds) = applyBassBoost(blues, greens, reds, numSamples, boost, lowPassCutoff, effectStart, effectEnd);

    //apply phase effect to data
    if (phaserEffect == 1)
        tie(outputBlues, outputGreens, outputReds) = applyPhaser(blues, greens, reds, numSamples, delay, feedback, dryWetMix, lowPassCutoff, oscillatorRate, frequencyRange, phaseShiftDepth, waveformType, effectStart, effectEnd);

    //echo is actually delay with feedback, set feedback to 0 to get pure delay effect
    //apply echo effect to data
    if (echoEffect == 1)
        tie(outputBlues, outputGreens, outputReds) = applyEcho(blues, greens, reds, numSamplesEffect, feedback, dryWetMix, decayTime, delayTime, effectStart, effectEnd, numSamples);

    if (pitchShiftEffect == 1)
        tie(outputBlues, outputGreens, outputReds) = changePitch(blues, greens, reds, numSamples, pitchShift, effectStart, effectEnd);

    //if (delayEffect == 1)
   // tie(outputBlues, outputGreens, outputReds) = applyEcho2(blues, greens, reds, numSamples, delay, decayFactor, dryWetMix );
    // Generate the output image data from modified data
    Vec3b* pixels = generateImagePixels(outputBlues, outputGreens, outputReds, numSamples, image.cols, image.rows);

    // Write the output image data to the output image file
    writeImageFile(outputImageFilename, pixels, image.cols, image.rows, effectDirection);

    return 0;
}

