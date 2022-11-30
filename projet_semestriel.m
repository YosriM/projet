%import and display image of a car
A=imread('voiture_7.jpeg');
A = imresize(A,0.25);
figure 
subplot(2,2,1);imshow(A);title('original');
Agray = rgb2gray(A);
subplot(2,2,2);imshow(Agray);title('gray');
%detect the car using ROI technology

%detect car color using simple segmentation
image=segmentImage(A);
subplot(2,2,3);imshow(image);title('segmented');
%detect and extract text of car number plate
text=ocr(A);
% Example 1 - Draw an ROI and evaluate OCR training
% ---------------------------------------------------
  I = imread('C:\Users\yousr\Desktop\voitures\taxi 1.jpeg');
  figure
  imshow(I)

%  % Draw a region of interest
  h = imrect
%
%  % Evaluate OCR within ROI
  roi = h.getPosition;
  ocrI = evaluateOCRTraining(I, roi);
%
%  % Show results
  figure
  imshow(ocrI)
%
%  Example 2 - Get all OCR results
%  -------------------------------
  I = imread('C:\Users\yousr\Desktop\voitures\taxi 1.jpeg');
  [ocrI, results] = evaluateOCRTraining(I);
  a = results.Text;
  b = results.CharacterConfidences;
%
%  Example 3 - Batch OCR evaluation
%  --------------------------------
%  % Save this generated function and use it in the imageBatchProcessor
%  imageBatchProcessor
%
% See also ocr, ocrTrainer, ocrText


