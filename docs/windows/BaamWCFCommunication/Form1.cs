using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.ServiceModel;

namespace BaamWCFTest
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

		private void button1_Click(object sender, EventArgs e)
		{

		}

		private void setValueBtn_Click(object sender, EventArgs e)
		{
			try
			{
				int command;
				if (int.TryParse(setValueCommand.Text, out command))
				{
					setValueReturn.Text = String.Empty;
					setValueReturn.Update();
					//Specify the binding to be used for the client.
					BasicHttpBinding binding = new BasicHttpBinding();

					//Specify the address to be used for the client.
					EndpointAddress addr = new EndpointAddress(String.Format(@"http://{0}:{1}{2}", server.Text, port.Text, address.Text));
					CIBaamInterfaceClient client = new CIBaamInterfaceClient(binding, addr);

					int result = client.SetValue(command, setValueData.Text);
					setValueReturn.Text = result.ToString();
					// Always close the client.
					client.Close();
				}
				else
					setValueReturn.Text = "Bad command ID";
			}
			catch (Exception ex)
			{
				setValueReturn.Text = ex.Message;
			}


		}

		private void getValueBtn_Click(object sender, EventArgs e)
		{
			try
			{
				int command;
				if (int.TryParse(getValueCommand.Text, out command))
				{
					getValueReturn.Text = String.Empty;
					getValueReturn.Update();
					//Specify the binding to be used for the client.
					BasicHttpBinding binding = new BasicHttpBinding();

					//Specify the address to be used for the client.
					EndpointAddress addr = new EndpointAddress(String.Format(@"http://{0}:{1}{2}", server.Text, port.Text, address.Text));
					CIBaamInterfaceClient client = new CIBaamInterfaceClient(binding, addr);

					string result = client.GetValue(command);
					getValueReturn.Text = result;
					// Always close the client.
					client.Close();
				}
				else
					getValueReturn.Text = "Bad command ID";
			}
			catch (Exception ex)
			{
				getValueReturn.Text = ex.Message;
			}


		}
	}
}
