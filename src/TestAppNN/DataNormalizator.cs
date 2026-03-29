using Skender.Stock.Indicators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestAppNN
{
    internal static class DataNormalizator
    {
        public static List<double> GetMovements(List<Quote> quotes)
        {
            return quotes.Select(x => (double)x.Close - (double)x.Open).ToList();
        }

        public static List<double> Normalize(IEnumerable<double> d)
        {
            var max = d.Select(x => Math.Abs(x)).Max();

            return d.Select(x => x / max).ToList();
        }
    }
}
