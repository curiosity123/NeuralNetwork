using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace NeuralNetworkPlayground
{
    public class WpfGraphics
    {
        BitmapSource bitmap;
        PixelFormat pf = PixelFormats.Rgb24;
        readonly int rawStride;
        byte[] pixelData;
        ObservableCollection<UIElement> canvasCollection;


        private WpfGraphics()
        {
            rawStride = (300 * pf.BitsPerPixel + 7) / 8;
            pixelData = new byte[rawStride * 300];
        }


        public WpfGraphics(ObservableCollection<UIElement> _canvasCollection):this()
        {
            canvasCollection = _canvasCollection;
        }





        public void SetPixel(int x, int y, byte r, byte g, byte b)
        {
            int xIndex = x * 3;
            int yIndex = y * rawStride;
            pixelData[xIndex + yIndex] = r;
            pixelData[xIndex + yIndex + 1] = g;
            pixelData[xIndex + yIndex + 2] = b;
        }
        public void Clear()
        {
            canvasCollection.Clear();
            pixelData = new byte[rawStride * 300];
        }


        public void Draw()
        {
            bitmap = BitmapSource.Create(300, 300, 96, 96, pf, null, pixelData, rawStride);
            Image img = new Image();

            img.Source = bitmap;

            canvasCollection.Add(img);

        }

    }
}
