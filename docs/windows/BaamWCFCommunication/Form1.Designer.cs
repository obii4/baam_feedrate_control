namespace BaamWCFTest
{
	partial class Form1
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.getValueBtn = new System.Windows.Forms.Button();
			this.label1 = new System.Windows.Forms.Label();
			this.getValueReturn = new System.Windows.Forms.Label();
			this.getValueCommand = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.server = new System.Windows.Forms.TextBox();
			this.setValueCommand = new System.Windows.Forms.TextBox();
			this.setValueData = new System.Windows.Forms.TextBox();
			this.setValueBtn = new System.Windows.Forms.Button();
			this.setValueReturn = new System.Windows.Forms.Label();
			this.label4 = new System.Windows.Forms.Label();
			this.address = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.port = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// getValueBtn
			// 
			this.getValueBtn.Location = new System.Drawing.Point(15, 78);
			this.getValueBtn.Name = "getValueBtn";
			this.getValueBtn.Size = new System.Drawing.Size(75, 23);
			this.getValueBtn.TabIndex = 0;
			this.getValueBtn.Text = "Get Value";
			this.getValueBtn.UseVisualStyleBackColor = true;
			this.getValueBtn.Click += new System.EventHandler(this.getValueBtn_Click);
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(12, 110);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(42, 13);
			this.label1.TabIndex = 1;
			this.label1.Text = "Return:";
			// 
			// getValueReturn
			// 
			this.getValueReturn.Location = new System.Drawing.Point(50, 110);
			this.getValueReturn.Name = "getValueReturn";
			this.getValueReturn.Size = new System.Drawing.Size(265, 51);
			this.getValueReturn.TabIndex = 2;
			this.getValueReturn.Text = "?????";
			// 
			// getValueCommand
			// 
			this.getValueCommand.Location = new System.Drawing.Point(96, 80);
			this.getValueCommand.Name = "getValueCommand";
			this.getValueCommand.Size = new System.Drawing.Size(45, 20);
			this.getValueCommand.TabIndex = 3;
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(12, 9);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(41, 13);
			this.label2.TabIndex = 4;
			this.label2.Text = "Server:";
			// 
			// server
			// 
			this.server.Location = new System.Drawing.Point(53, 6);
			this.server.Name = "server";
			this.server.Size = new System.Drawing.Size(75, 20);
			this.server.TabIndex = 5;
			this.server.Text = "localhost";
			// 
			// setValueCommand
			// 
			this.setValueCommand.Location = new System.Drawing.Point(96, 166);
			this.setValueCommand.Name = "setValueCommand";
			this.setValueCommand.Size = new System.Drawing.Size(45, 20);
			this.setValueCommand.TabIndex = 6;
			// 
			// setValueData
			// 
			this.setValueData.Location = new System.Drawing.Point(147, 166);
			this.setValueData.Name = "setValueData";
			this.setValueData.Size = new System.Drawing.Size(168, 20);
			this.setValueData.TabIndex = 7;
			// 
			// setValueBtn
			// 
			this.setValueBtn.Location = new System.Drawing.Point(15, 164);
			this.setValueBtn.Name = "setValueBtn";
			this.setValueBtn.Size = new System.Drawing.Size(75, 23);
			this.setValueBtn.TabIndex = 8;
			this.setValueBtn.Text = "Set Value";
			this.setValueBtn.UseVisualStyleBackColor = true;
			this.setValueBtn.Click += new System.EventHandler(this.setValueBtn_Click);
			// 
			// setValueReturn
			// 
			this.setValueReturn.Location = new System.Drawing.Point(60, 197);
			this.setValueReturn.Name = "setValueReturn";
			this.setValueReturn.Size = new System.Drawing.Size(265, 46);
			this.setValueReturn.TabIndex = 10;
			this.setValueReturn.Text = "?????";
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point(12, 197);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(42, 13);
			this.label4.TabIndex = 9;
			this.label4.Text = "Return:";
			// 
			// address
			// 
			this.address.Location = new System.Drawing.Point(63, 37);
			this.address.Name = "address";
			this.address.Size = new System.Drawing.Size(286, 20);
			this.address.TabIndex = 12;
			this.address.Text = "/Design_Time_Addresses/BaamHmi/CIBaamInterface/";
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(12, 40);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(48, 13);
			this.label3.TabIndex = 11;
			this.label3.Text = "Address:";
			// 
			// port
			// 
			this.port.Location = new System.Drawing.Point(184, 6);
			this.port.Name = "port";
			this.port.Size = new System.Drawing.Size(75, 20);
			this.port.TabIndex = 14;
			this.port.Text = "8733";
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Location = new System.Drawing.Point(143, 9);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(29, 13);
			this.label5.TabIndex = 13;
			this.label5.Text = "Port:";
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(361, 241);
			this.Controls.Add(this.port);
			this.Controls.Add(this.label5);
			this.Controls.Add(this.address);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.setValueReturn);
			this.Controls.Add(this.label4);
			this.Controls.Add(this.setValueBtn);
			this.Controls.Add(this.setValueData);
			this.Controls.Add(this.setValueCommand);
			this.Controls.Add(this.server);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.getValueCommand);
			this.Controls.Add(this.getValueReturn);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.getValueBtn);
			this.Name = "Form1";
			this.Text = "CINCINNATI BAAM WCF Test";
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button getValueBtn;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label getValueReturn;
		private System.Windows.Forms.TextBox getValueCommand;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox server;
		private System.Windows.Forms.TextBox setValueCommand;
		private System.Windows.Forms.TextBox setValueData;
		private System.Windows.Forms.Button setValueBtn;
		private System.Windows.Forms.Label setValueReturn;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox address;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox port;
		private System.Windows.Forms.Label label5;
	}
}

