using NeuralNetworkScratch;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNetworkPlayground
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NEngine nn;
        List<Point> BluePoint = new List<Point>();
        List<Point> OrangePoint = new List<Point>();

        public MainWindow()
        {
            InitializeComponent();


        }



        BitmapSource bitmap;
        PixelFormat pf = PixelFormats.Rgb24;
        int rawStride;
        byte[] pixelData;

        void SetPixel(int x, int y, byte c, byte[] buffer, int rawStride)
        {
            int xIndex = x * 3;
            int yIndex = y * rawStride;
            buffer[xIndex + yIndex] = c;
            buffer[xIndex + yIndex + 1] = c;
            buffer[xIndex + yIndex + 2] = c;
        }

        private void Draw()
        {
            ClearCanvas();

            rawStride = (400 * pf.BitsPerPixel + 7) / 8;
            pixelData = new byte[rawStride * 400];


            for (int x = 0; x < 400; x++)
                for (int y = 0; y < 400; y++)
                {
                    double[,] input = new double[,] { { ((double)x / 400), ((double)y / 400) } };
                    double[,] result = nn.CheckAnswer(input);
                    //if (result[0, 0] > 0)
                    SetPixel(x, y, (byte)(result[0, 0] * 254), pixelData, rawStride);
                    //else
                    //    SetPixel(x, y, Colors.Green, pixelData, rawStride);
                }


            bitmap = BitmapSource.Create(400, 400, 96, 96, pf, null, pixelData, rawStride);
            Image img = new Image();
            img.Source = bitmap;

            canvas.Children.Add(img);

            DrawPoints();

        }

        private void DrawPoints()
        {
            foreach (Point p in BluePoint)
                DrawPoint(p.X, p.Y, Colors.Blue);
            foreach (Point p in OrangePoint)
                DrawPoint(p.X, p.Y, Colors.OrangeRed);
        }

        private void DrawPoint(double x, double y, Color c)
        {
            Ellipse el = new Ellipse();
            Canvas.SetTop(el, y - 4);
            Canvas.SetLeft(el, x - 4);
            el.Width = 4;
            el.Height = 4;
            el.Fill = new SolidColorBrush(c);
            canvas.Children.Add(el);
        }

        private void ClearCanvas()
        {
            canvas.Children.Clear();
        }

        private void Clear_Click(object sender, RoutedEventArgs e)
        {
            ClearCanvas();
            BluePoint.Clear();
            OrangePoint.Clear();
        }

        private void Lear_Click(object sender, RoutedEventArgs e)
        {
            if (nn != null)
            {
                nn.BackwardPropagation();
                Draw();
            }
        }


        private void canvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            Point p = e.GetPosition(this);
            if (e.RightButton == MouseButtonState.Pressed)
            {
                DrawPoint(p.X, p.Y, Colors.BlueViolet);
                BluePoint.Add(p);
            }
            else
            {
                DrawPoint(p.X, p.Y, Colors.Orange);
                OrangePoint.Add(p);
            }

            DrawPoints();

            double[,] X = new double[BluePoint.Count() + OrangePoint.Count(), 2];
            double[,] Y = new double[BluePoint.Count() + OrangePoint.Count(), 1];

            for (int i = 0; i < BluePoint.Count(); i++)
            {
                X[i, 0] = BluePoint[i].X / 400;
                X[i, 1] = BluePoint[i].Y / 400;
                Y[i, 0] = 1;
            }

            for (int i = BluePoint.Count(); i < BluePoint.Count() + OrangePoint.Count(); i++)
            {
                X[i, 0] = OrangePoint[i - BluePoint.Count()].X / 400;
                X[i, 1] = OrangePoint[i - BluePoint.Count()].Y / 400;
                Y[i, 0] = -1;
            }


            Layer[] layers = new Layer[]
            {
                new Layer(LayerType.Input,  X, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 4, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 4, ActivationFunction.Tanh),
                               new Layer(LayerType.Hidden, 3, ActivationFunction.Tanh),
                new Layer(LayerType.Output, Y, ActivationFunction.Tanh)
            };
            nn = new NEngine(layers, Y);
            NeuralNetworkScratch.Matrix.Print(nn.ForwardPropagation());

        }


        private void CheckLoss_Click(object sender, RoutedEventArgs e)
        {
              NeuralNetworkScratch.Matrix.Print(nn.ForwardPropagation());
        }

        private void Print_Click(object sender, RoutedEventArgs e)
        {
            Draw();
        }
    }
}
