using NeuralNetworkScratch;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interactivity;

namespace NeuralNetworkPlayground
{
    public class MainWindowViewModel : INotifyPropertyChanged
    {

        NEngine nn;
        WpfGraphics wpfGraphics;
        List<Point> BluePoint = new List<Point>();
        List<Point> OrangePoint = new List<Point>();
        bool IsLearning = false;

        public MainWindowViewModel()
        {
            wpfGraphics = new WpfGraphics(CanvasCollection);
            Topology = "6;5;4";
            LayersParser(Topology);
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
            IsLearning = true;
            Task.Factory.StartNew(Learning);
        }

        public ICommand StopLearnCommand { get { return new RelayCommand(x => true, Stop); } }
        private void Stop(object obj)
        {
            IsLearning = false;
        }


        private void Learning()
        {
            while (IsLearning)
                if (nn != null)
                {

                    nn.BackwardPropagation(1000);

                    Application.Current.Dispatcher.Invoke((Action)(() =>
                {
                    DrawNetworkAnswer();
                    LossRMSE = "RMSE:" + nn.GetRMSELoss(X2, Y2);
                    LossRMSETest = "RMSE Test:" + nn.GetRMSELoss(TestX, TestY);
                }));

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


        private double learningRate =0.1;

        public double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value;
                RaisePropertyChangedEvent("LearningRate");
            }
        }

        private double learnToTest=0.8;

        public double LearnToTest
        {
            get { return learnToTest; }
            set
            {
                learnToTest = value;
                RaisePropertyChangedEvent("LearnToTest");
            }
        }


        public ICommand TopologyChangedCommand { get { return new RelayCommand(x => true, TopologyChanged); } }
        private void TopologyChanged(object obj)
        {
            LayersParser(obj);
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
            IsLearning = false;
            wpfGraphics.Clear();
            BluePoint.Clear();
            OrangePoint.Clear();

        }

        double[,] X2;
        double[,] TestX;
        double[,] Y2;
        double[,] TestY;
        List<Layer> HiddenLayers;


        private void AddNewPoint(MouseButton mb)
        {
            IsLearning = false;
            Point p = new Point(PanelX, PanelY);
            if (mb == MouseButton.Left)
                BluePoint.Add(p);
            else
                OrangePoint.Add(p);

            wpfGraphics.Clear();

            DrawPoints();

            double[,] X = new double[BluePoint.Count() + OrangePoint.Count(), 2];
            double[,] Y = new double[BluePoint.Count() + OrangePoint.Count(), 2];

            for (int i = 0; i < BluePoint.Count(); i++)
            {
                X[i, 0] = BluePoint[i].X / 300;
                X[i, 1] = BluePoint[i].Y / 300;
                Y[i, 0] = 1;
                Y[i, 1] = -1;
            }

            for (int i = BluePoint.Count(); i < BluePoint.Count() + OrangePoint.Count(); i++)
            {
                X[i, 0] = (OrangePoint[i - BluePoint.Count()].X / 300);
                X[i, 1] = (OrangePoint[i - BluePoint.Count()].Y / 300);
                Y[i, 0] = -1;
                Y[i, 1] = 1;
            }
            NeuralNetworkScratch.Matrix.Unsort(ref X, ref Y, new Random());

            NeuralNetworkScratch.Matrix.SplitMatrix(X, out X2, out TestX, LearnToTest);
            NeuralNetworkScratch.Matrix.SplitMatrix(Y, out Y2, out TestY, LearnToTest);


            Layer[] layers = new Layer[2 + HiddenLayers.Count()];

            layers[0] = new Layer(LayerType.Input, X2, ActivationFunction.Tanh);

            for (int i = 1; i < HiddenLayers.Count() + 1; i++)
                layers[i] = HiddenLayers[i - 1];

            layers[layers.Count() - 1] = new Layer(LayerType.Output, Y2, ActivationFunction.Tanh);

            nn = new NEngine(layers, Y2,LearningRate, 0);

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
                    //if (result[0, 0] >= 0)
                    //    color = (byte)(127 + (byte)(result[0, 0] * 127));
                    //else
                    //    color = (byte)(127 + (byte)(result[0, 0] * 127));
                    if (result[0, 0] > 0)
                    {
                        color = (byte)(127 + (byte)(result[0, 0] * 127));
                        wpfGraphics.SetPixel(x, y, 0, color, 0);
                    }
                    else
                    {
                        color = (byte)(127 + (byte)(result[0, 1] * 127));
                        wpfGraphics.SetPixel(x, y, color, color, 0);
                    }
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
