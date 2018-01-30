using NeuralNetworkScratch;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interactivity;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace NeuralNetworkPlayground
{
    public class MainWindowViewModel : INotifyPropertyChanged
    {

        private double _panelX;
        private double _panelY;

        public double PanelX
        {
            get { return _panelX; }
            set
            {
                if (value.Equals(_panelX)) return;
                _panelX = value;
                RaisePropertyChangedEvent("PanelX");
            }
        }

        public double PanelY
        {
            get { return _panelY; }
            set
            {
                if (value.Equals(_panelY)) return;
                _panelY = value;
                RaisePropertyChangedEvent("PanelY");
            }
        }


        private ObservableCollection<UIElement> myVar = new ObservableCollection<UIElement>();

        public ObservableCollection<UIElement> CanvasCollection
        {
            get { return myVar; }
            set { myVar = value; }
        }


        private void MouseClick(MouseButton mb)
        {
            nn = null;
            Point p = new Point(PanelX, PanelY);
            if (mb == MouseButton.Left)
                BluePoint.Add(p);
            else
                OrangePoint.Add(p);

            Draw();

            double[,] X = new double[BluePoint.Count() + OrangePoint.Count(), 2];
            double[,] Y = new double[BluePoint.Count() + OrangePoint.Count(), 1];

            for (int i = 0; i < BluePoint.Count(); i++)
            {
                X[i, 0] = BluePoint[i].X / 300;
                X[i, 1] = BluePoint[i].Y / 300;
                Y[i, 0] = 1;
            }

            for (int i = BluePoint.Count(); i < BluePoint.Count() + OrangePoint.Count(); i++)
            {
                X[i, 0] = (OrangePoint[i - BluePoint.Count()].X / 300);
                X[i, 1] = (OrangePoint[i - BluePoint.Count()].Y / 300);
                Y[i, 0] = -1;
            }


            Layer[] layers = new Layer[]
            {
                new Layer(LayerType.Input,  X, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 6, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 5, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 4, ActivationFunction.Tanh),
                new Layer(LayerType.Output, Y, ActivationFunction.Tanh)
            };
            nn = new NEngine(layers, Y, 0.2, 0.001);
            NeuralNetworkScratch.Matrix.Print(nn.ForwardPropagation());
        }


        public ICommand MouseLeftCommand { get { return new RelayCommand(x => true, MouseLeft); } }
        private void MouseLeft(object obj)
        {
            MouseClick(MouseButton.Left);

        }


        public ICommand MouseRightCommand { get { return new RelayCommand(x => true, MouseRight); } }
        private void MouseRight(object obj)
        {
            MouseClick(MouseButton.Right);
        }


        public ICommand LearnCommand { get { return new RelayCommand(x => true, Learn); } }
        private void Learn(object obj)
        {
            if (nn != null)
            {
                nn.BackwardPropagation(500);
                Draw();
            }
        }


        public ICommand ClearCommand { get { return new RelayCommand(x => true, Clear); } }
        private void Clear(object obj)
        {
            ClearCanvas();
            BluePoint.Clear();
            OrangePoint.Clear();
        }





        NEngine nn;
        List<Point> BluePoint = new List<Point>();
        List<Point> OrangePoint = new List<Point>();

        BitmapSource bitmap;
        PixelFormat pf = PixelFormats.Rgb24;
        int rawStride;
        byte[] pixelData;

        void SetPixel(int x, int y, byte r, byte g, byte b, byte[] buffer, int rawStride)
        {
            int xIndex = x * 3;
            int yIndex = y * rawStride;
            buffer[xIndex + yIndex] = r;
            buffer[xIndex + yIndex + 1] = g;
            buffer[xIndex + yIndex + 2] = b;
        }

        private void Draw()
        {
            ClearCanvas();

            rawStride = (300 * pf.BitsPerPixel + 7) / 8;
            pixelData = new byte[rawStride * 300];

            if (nn != null)
                for (int x = 0; x < 300; x++)
                    for (int y = 0; y < 300; y++)
                    {

                        double[,] input = new double[,] { { ((double)x / 300), ((double)y / 300) } };
                        double[,] result = nn.CheckAnswer(input);

                        byte color = 0;
                        if (result[0, 0] >= 0)
                            color = (byte)(127 + (byte)(result[0, 0] * 127));
                        else
                            color = (byte)(127 + (byte)(result[0, 0] * 127));

                        SetPixel(x, y, 0, color, 0, pixelData, rawStride);
                    }

            foreach (Point p in BluePoint)
            {
                SetPixel((int)p.X, (int)p.Y, 0, 0, 255, pixelData, rawStride);
                SetPixel((int)p.X + 1, (int)p.Y, 0, 0, 255, pixelData, rawStride);
                SetPixel((int)p.X, (int)p.Y + 1, 0, 0, 255, pixelData, rawStride);
                SetPixel((int)p.X + 1, (int)p.Y + 1, 0, 0, 255, pixelData, rawStride);
            }
            foreach (Point p in OrangePoint)
            {
                SetPixel((int)p.X, (int)p.Y, 255, 0, 0, pixelData, rawStride);
                SetPixel((int)p.X + 1, (int)p.Y, 255, 0, 0, pixelData, rawStride);
                SetPixel((int)p.X, (int)p.Y + 1, 255, 0, 0, pixelData, rawStride);
                SetPixel((int)p.X + 1, (int)p.Y + 1, 255, 0, 0, pixelData, rawStride);
            }


            bitmap = BitmapSource.Create(300, 300, 96, 96, pf, null, pixelData, rawStride);
            Image img = new Image();

            img.Source = bitmap;

            CanvasCollection.Add(img);
        }

        private void ClearCanvas()
        {
            CanvasCollection.Clear();
        }


        public event PropertyChangedEventHandler PropertyChanged;
        protected void RaisePropertyChangedEvent(string propertyName)
        {
            var handler = PropertyChanged;
            if (handler != null)
                handler(this, new PropertyChangedEventArgs(propertyName));
        }

    }
    public class MouseBehaviour : Behavior<ItemsControl>
    {
        public static readonly DependencyProperty MouseYProperty = DependencyProperty.Register(
           "MouseY", typeof(double), typeof(MouseBehaviour), new PropertyMetadata(default(double)));

        public static readonly DependencyProperty MouseXProperty = DependencyProperty.Register(
           "MouseX", typeof(double), typeof(MouseBehaviour), new PropertyMetadata(default(double)));

        public double MouseY
        {
            get { return (double)GetValue(MouseYProperty); }
            set { SetValue(MouseYProperty, value); }
        }

        public double MouseX
        {
            get { return (double)GetValue(MouseXProperty); }
            set { SetValue(MouseXProperty, value); }
        }

        protected override void OnAttached()
        {
            AssociatedObject.MouseMove += AssociatedObjectOnMouseMove;
        }

        private void AssociatedObjectOnMouseMove(object sender, MouseEventArgs mouseEventArgs)
        {
            var pos = mouseEventArgs.GetPosition(AssociatedObject);
            MouseX = pos.X;
            MouseY = pos.Y;
        }

        protected override void OnDetaching()
        {
            AssociatedObject.MouseMove -= AssociatedObjectOnMouseMove;
        }
    }

    public class RelayCommand : ICommand
    {
        private Predicate<object> _canExecute;
        private Action<object> _execute;

        public RelayCommand(Predicate<object> canExecute, Action<object> execute)
        {
            this._canExecute = canExecute;
            this._execute = execute;
        }

        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object parameter)
        {
            return _canExecute(parameter);
        }

        public void Execute(object parameter)
        {
            _execute(parameter);
        }
    }
}
