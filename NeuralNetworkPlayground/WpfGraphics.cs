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
        int rawStride;
        byte[] pixelData;
        ObservableCollection<UIElement> canvasCollection;
        int Width, Height;

        private WpfGraphics()
        {

        }


        public WpfGraphics(ObservableCollection<UIElement> _canvasCollection, int _width = 300, int _height = 300)
        {
            Width = _width;
            Height = _height;
            rawStride = (Height * pf.BitsPerPixel + 7) / 8;
            pixelData = new byte[rawStride * Width];
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
            pixelData = new byte[rawStride * Height];
        }
        public void Draw()
        {
            bitmap = BitmapSource.Create(Width, Height, 96, 96, pf, null, pixelData, rawStride);
            Image img = new Image();

            img.Source = bitmap;

            canvasCollection.Add(img);

        }
    }
}
