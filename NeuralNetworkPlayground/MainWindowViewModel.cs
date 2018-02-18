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

        NEngine nn;
        WpfGraphics wpfGraphics;
        List<Point> BluePoint = new List<Point>();
        List<Point> OrangePoint = new List<Point>();

        public MainWindowViewModel()
        {
            wpfGraphics = new WpfGraphics(CanvasCollection);
        }

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


        private ObservableCollection<UIElement> canvasCollection = new ObservableCollection<UIElement>();

        public ObservableCollection<UIElement> CanvasCollection
        {
            get { return canvasCollection; }
            set { canvasCollection = value; }
        }

        public ICommand MouseLeftCommand { get { return new RelayCommand(x => true, MouseLeft); } }
        private void MouseLeft(object obj)
        {
            AddNewPoint(MouseButton.Left);

        }


        public ICommand MouseRightCommand { get { return new RelayCommand(x => true, MouseRight); } }
        private void MouseRight(object obj)
        {
            AddNewPoint(MouseButton.Right);
        }


        public ICommand LearnCommand { get { return new RelayCommand(x => true, Learn); } }
        private void Learn(object obj)
        {
            if (nn != null)
            {
                nn.BackwardPropagation(500);
                DrawNetworkAnswer();
            }
        }


        public ICommand ClearCommand { get { return new RelayCommand(x => true, Clear); } }
        private void Clear(object obj)
        {
            wpfGraphics.Clear();
            BluePoint.Clear();
            OrangePoint.Clear();

        }


        private void AddNewPoint(MouseButton mb)
        {
            Point p = new Point(PanelX, PanelY);
            if (mb == MouseButton.Left)
                BluePoint.Add(p);
            else
                OrangePoint.Add(p);

            wpfGraphics.Clear();

            DrawPoints();

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

            double[,] X2;
            double[,] Test2;
            NeuralNetworkScratch.Matrix.SplitMatrix(X,out X2,out Test2,0.7);

            Layer[] layers = new Layer[]
            {
                new Layer(LayerType.Input,  X, ActivationFunction.Sigmoid),
                new Layer(LayerType.Hidden, 7, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 6, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 5, ActivationFunction.Tanh),
                new Layer(LayerType.Output, Y, ActivationFunction.Tanh)
            };
            nn = new NEngine(layers, Y, 0.1, 0);

            NeuralNetworkScratch.Matrix.Print(nn.ForwardPropagation());
        }

        private void DrawNetworkAnswer()
        {
            wpfGraphics.Clear();
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

                    wpfGraphics.SetPixel(x, y, 0, color, 0);
                }

            DrawPoints();
        }

        private void DrawPoints()
        {

            foreach (Point p in BluePoint)
            {
                wpfGraphics.SetPixel((int)p.X, (int)p.Y, 0, 0, 255);
                wpfGraphics.SetPixel((int)p.X + 1, (int)p.Y, 0, 0, 255);
                wpfGraphics.SetPixel((int)p.X, (int)p.Y + 1, 0, 0, 255);
                wpfGraphics.SetPixel((int)p.X + 1, (int)p.Y + 1, 0, 0, 255);
            }
            foreach (Point p in OrangePoint)
            {
                wpfGraphics.SetPixel((int)p.X, (int)p.Y, 255, 0, 0);
                wpfGraphics.SetPixel((int)p.X + 1, (int)p.Y, 255, 0, 0);
                wpfGraphics.SetPixel((int)p.X, (int)p.Y + 1, 255, 0, 0);
                wpfGraphics.SetPixel((int)p.X + 1, (int)p.Y + 1, 255, 0, 0);
            }
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
