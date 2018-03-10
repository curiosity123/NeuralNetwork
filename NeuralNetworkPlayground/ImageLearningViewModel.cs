using Microsoft.Win32;
using NeuralNetworkScratch;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interactivity;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace NeuralNetworkPlayground
{
    public class ImageLearningViewModel : INotifyPropertyChanged
    {

        NEngine nn;
        WpfGraphics wpfGraphics;
        bool IsLearningRightNow = false;

        public ImageLearningViewModel()
        {
            wpfGraphics = new WpfGraphics(CanvasCollection2, 100, 100);
            Topology = "15;15;15;15";
            LayersParser(Topology);
        }

        private ObservableCollection<UIElement> canvasCollection = new ObservableCollection<UIElement>();

        public ObservableCollection<UIElement> CanvasCollection2
        {
            get { return canvasCollection; }
            set { canvasCollection = value; }
        }


        public ICommand LearnCommand { get { return new RelayCommand(x => true, Learn); } }
        private void Learn(object obj)
        {
            LearningCommandManager();
        }

        private void LearningCommandManager()
        {
            if (!IsLearningRightNow)
            {
                IsLearningRightNow = true;
                Task.Factory.StartNew(Learning);
            }
            else
            {
                IsLearningRightNow = false;
            }
        }

        private void Learning()
        {
            int i = 0;
            while (IsLearningRightNow)
                if (nn != null)
                {
                    nn.BackwardPropagation(1);
                    Application.Current.Dispatcher.Invoke((Action)(() =>
                {
                    if (i % 10 == 0)
                        DrawNetworkAnswer();
                    LossRMSE ="Epoch:"+ i.ToString() + " RMSE:" + nn.GetRMSELoss(X, Y);
                    i++;
                    ButtonLearnTitle = "Stop learning process";
                }));

                }
            ButtonLearnTitle = "Start learning process";
        }

        private string buttonLearnTitle = "Start learning process";

        public string ButtonLearnTitle
        {
            get { return buttonLearnTitle; }
            set
            {
                buttonLearnTitle = value;
                RaisePropertyChangedEvent("ButtonLearnTitle");
            }
        }



        private string lossRMSETest = "RMSE Test: 0";

        public string LossRMSETest
        {
            get { return lossRMSETest; }
            set
            {
                lossRMSETest = value;
                RaisePropertyChangedEvent("LossRMSETest");
            }
        }


        private string lossRMSE = "RMSE: 0";

        public string LossRMSE
        {
            get { return lossRMSE; }
            set
            {
                lossRMSE = value;
                RaisePropertyChangedEvent("LossRMSE");
            }
        }

        private string topology;

        public string Topology
        {
            get { return topology; }
            set
            {
                topology = value;
                RaisePropertyChangedEvent("Topology");
            }
        }


        private double learningRate = 0.1;

        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                learningRate = value;
                RaisePropertyChangedEvent("LearningRate");
            }
        }

        private double learnToTest = 0.8;

        public double LearnToTest
        {
            get { return learnToTest; }
            set
            {
                learnToTest = value;
                RaisePropertyChangedEvent("LearnToTest");
            }
        }


        private BitmapImage testBitmap;

        public BitmapImage TestBitmap
        {
            get { return testBitmap; }
            set
            {
                testBitmap = value;
                RaisePropertyChangedEvent("TestBitmap");
            }
        }


        public ICommand TopologyChangedCommand { get { return new RelayCommand(x => true, TopologyChanged); } }
        private void TopologyChanged(object obj)
        {
            LayersParser(obj);
        }

        public ICommand OpenCommand { get { return new RelayCommand(x => true, Open); } }
        private void Open(object obj)
        { 

            string Path = "";
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            openFileDialog1.InitialDirectory = Environment.CurrentDirectory;
            openFileDialog1.Filter = "Image files (*.jpg, *.jpeg, *.jpe, *.jfif, *.png) | *.jpg; *.jpeg; *.jpe; *.jfif; *.png";
            if (openFileDialog1.ShowDialog() == true)
                Path = openFileDialog1.FileName;
            else
                return;

            Bitmap original = (Bitmap)System.Drawing.Image.FromFile(Path);
            Bitmap resized = new Bitmap(original, new System.Drawing.Size(100,100));
            TestBitmap = BitmapToBitmapImage(resized);

           // TestBitmap = new BitmapImage(new Uri(Path));


            InitializeBitmapData();
            //open image
        }

        private BitmapImage BitmapToBitmapImage(Bitmap b)
        {
            MemoryStream ms = new MemoryStream();
            b.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
            BitmapImage image = new BitmapImage();
            image.BeginInit();
            ms.Seek(0, SeekOrigin.Begin);
            image.StreamSource = ms;
            image.EndInit();
            return image;
        }

        public static System.Windows.Media.Color GetPixelColor(BitmapSource bitmap, int x, int y)
        {
            System.Windows.Media.Color color;
            var bytesPerPixel = (bitmap.Format.BitsPerPixel + 7) / 8;
            var bytes = new byte[bytesPerPixel];
            var rect = new Int32Rect(x, y, 1, 1);

            bitmap.CopyPixels(rect, bytes, bytesPerPixel, 0);

            if (bitmap.Format == PixelFormats.Bgra32)
            {
                color = System.Windows.Media.Color.FromArgb(bytes[3], bytes[2], bytes[1], bytes[0]);
            }
            else if (bitmap.Format == PixelFormats.Bgr32)
            {
                color = System.Windows.Media.Color.FromRgb(bytes[2], bytes[1], bytes[0]);
            }
            // handle other required formats
            else
            {
                color = Colors.Black;
            }

            return color;
        }

        private void LayersParser(object obj)
        {
            string[] layers = (obj as string).Split(';');
            HiddenLayers = new List<Layer>();

            int value = 0;
            foreach (string s in layers)
                if (int.TryParse(s, out value) && value > 0)
                    HiddenLayers.Add(new Layer(LayerType.Hidden, value, ActivationFunction.Tanh));

            if (HiddenLayers.Count < 1)
            {
                Topology = "4;3";
                LayersParser(Topology);
            }
        }

        public ICommand ClearCommand { get { return new RelayCommand(x => true, Clear); } }
        private void Clear(object obj)
        {
            IsLearningRightNow = false;
            wpfGraphics.Clear();
        }

        List<Layer> HiddenLayers;

        double[,] X;
        double[,] Y;

        private byte getByte(double b)
        {
            if (b < -1)
                b = -1;
            if (b > 1)
                b = 1;

            return (byte)(((b + 1) / 2) * 255);
        }

        private double getDouble(byte b)
        {
            double result = 0;
            result = (double)(b - 128) / 256;
            result *= 2;

            return result;
        }


        private void InitializeBitmapData()
        {
            IsLearningRightNow = false;


            wpfGraphics.Clear();


            X = new double[10000, 2];
            Y = new double[10000, 3];


            for (int x = 0; x < 100; x++)
                for (int y = 0; y < 100; y++)
                {
                    X[x * 100 + y, 0] = (double)(((double)(x - 50)) / 50);
                    X[x * 100 + y, 1] = (double)(((double)(y - 50)) / 50);
                    Y[x * 100 + y, 0] = getDouble((GetPixelColor(TestBitmap, x, y).R));
                    Y[x * 100 + y, 1] = getDouble((GetPixelColor(TestBitmap, x, y).G));
                    Y[x * 100 + y, 2] = getDouble((GetPixelColor(TestBitmap, x, y).B));
                }



            //NeuralNetworkScratch.Matrix.Unsort(ref X, ref Y, new Random());

            Layer[] layers = new Layer[2 + HiddenLayers.Count()];

            layers[0] = new Layer(LayerType.Input, X, ActivationFunction.Tanh);

            for (int i = 1; i < HiddenLayers.Count() + 1; i++)
                layers[i] = HiddenLayers[i - 1];

            layers[layers.Count() - 1] = new Layer(LayerType.Output, Y, ActivationFunction.Tanh);

            nn = new NEngine(layers, Y, LearningRate, 0);

            NeuralNetworkScratch.Matrix.Print(nn.ForwardPropagation());
        }

        private void DrawNetworkAnswer()
        {
            wpfGraphics.Clear();
            Parallel.For(0, 100, x =>
            {
                Parallel.For(0, 100, y =>
                {
                    double[,] input = new double[1, 2];
                    input[0, 0] = (double)(((double)(x - 50)) / 50);
                    input[0, 1] = (double)(((double)(y - 50)) / 50);
                    double[,] result = nn.CheckAnswer(input);
                    wpfGraphics.SetPixel(x, y, getByte(result[0, 0]), getByte(result[0, 1]), getByte(result[0, 2]));// getByte(result[0, 1]), getByte(result[0, 2]));
                });
            });
            wpfGraphics.Draw();

        }



        public event PropertyChangedEventHandler PropertyChanged;
        protected void RaisePropertyChangedEvent(string propertyName)
        {
            var handler = PropertyChanged;
            if (handler != null)
                handler(this, new PropertyChangedEventArgs(propertyName));
        }
    }

}
