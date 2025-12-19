#pragma once

#include <msclr/gcroot.h>
#include "ImageProcessorWrapper.h"
#include <chrono>

namespace pavicgui2024 {

    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Windows::Forms;
    using namespace System::Drawing;
    using namespace std::chrono;

    public ref class Pavic_gui_2024_Form : public System::Windows::Forms::Form
    {
    public:
        Pavic_gui_2024_Form(void)
        {
            InitializeComponent();
            processor = gcnew ImageProcessorWrapper();
        }

    protected:
        ~Pavic_gui_2024_Form()
        {
            if (components)
            {
                delete components;
            }
        }

    private:
        ImageProcessorWrapper^ processor;
        System::ComponentModel::Container ^components;

        // UI controls
        System::Windows::Forms::Button^ btOpen;
        System::Windows::Forms::PictureBox^ pbox_input;
        System::Windows::Forms::PictureBox^ pbox_output;
        System::Windows::Forms::TextBox^ textB_Time;

        System::Windows::Forms::Button^ btGaussian_CPU;
        System::Windows::Forms::Button^ btEdge_CPU;
        System::Windows::Forms::Button^ btSharpen_CPU;

        System::Windows::Forms::Button^ btGaussian_CUDA;
        System::Windows::Forms::Button^ btEdge_CUDA;
        System::Windows::Forms::Button^ btSharpen_CUDA;

        void InitializeComponent(void)
        {
            this->components = gcnew System::ComponentModel::Container();
            this->Text = L"PROJECT: PAVIC LAB 2025";
            this->ClientSize = System::Drawing::Size(1000, 700);

            // Open button
            this->btOpen = (gcnew System::Windows::Forms::Button());
            this->btOpen->Location = System::Drawing::Point(10, 10);
            this->btOpen->Size = System::Drawing::Size(100, 30);
            this->btOpen->Text = L"Open";
            this->btOpen->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btOpen_Click);
            this->Controls->Add(this->btOpen);

            // Input picture box
            this->pbox_input = (gcnew System::Windows::Forms::PictureBox());
            this->pbox_input->Location = System::Drawing::Point(10, 50);
            this->pbox_input->Size = System::Drawing::Size(450, 450);
            this->pbox_input->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
            this->pbox_input->SizeMode = PictureBoxSizeMode::StretchImage;
            this->Controls->Add(this->pbox_input);

            // Output picture box
            this->pbox_output = (gcnew System::Windows::Forms::PictureBox());
            this->pbox_output->Location = System::Drawing::Point(480, 50);
            this->pbox_output->Size = System::Drawing::Size(450, 450);
            this->pbox_output->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
            this->pbox_output->SizeMode = PictureBoxSizeMode::StretchImage;
            this->Controls->Add(this->pbox_output);

            // Time textbox
            this->textB_Time = (gcnew System::Windows::Forms::TextBox());
            this->textB_Time->Location = System::Drawing::Point(10, 520);
            this->textB_Time->Size = System::Drawing::Size(300, 23);
            this->textB_Time->ReadOnly = true;
            this->Controls->Add(this->textB_Time);

            // CPU buttons
            this->btGaussian_CPU = (gcnew System::Windows::Forms::Button());
            this->btGaussian_CPU->Location = System::Drawing::Point(10, 560);
            this->btGaussian_CPU->Size = System::Drawing::Size(150, 30);
            this->btGaussian_CPU->Text = L"Gaussian - CPU";
            this->btGaussian_CPU->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btGaussian_CPU_Click);
            this->Controls->Add(this->btGaussian_CPU);

            this->btEdge_CPU = (gcnew System::Windows::Forms::Button());
            this->btEdge_CPU->Location = System::Drawing::Point(170, 560);
            this->btEdge_CPU->Size = System::Drawing::Size(150, 30);
            this->btEdge_CPU->Text = L"Edge - CPU";
            this->btEdge_CPU->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btEdge_CPU_Click);
            this->Controls->Add(this->btEdge_CPU);

            this->btSharpen_CPU = (gcnew System::Windows::Forms::Button());
            this->btSharpen_CPU->Location = System::Drawing::Point(330, 560);
            this->btSharpen_CPU->Size = System::Drawing::Size(150, 30);
            this->btSharpen_CPU->Text = L"Sharpen - CPU";
            this->btSharpen_CPU->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btSharpen_CPU_Click);
            this->Controls->Add(this->btSharpen_CPU);

            // CUDA buttons
            this->btGaussian_CUDA = (gcnew System::Windows::Forms::Button());
            this->btGaussian_CUDA->Location = System::Drawing::Point(480, 560);
            this->btGaussian_CUDA->Size = System::Drawing::Size(150, 30);
            this->btGaussian_CUDA->Text = L"Gaussian - CUDA";
            this->btGaussian_CUDA->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btGaussian_CUDA_Click);
            this->Controls->Add(this->btGaussian_CUDA);

            this->btEdge_CUDA = (gcnew System::Windows::Forms::Button());
            this->btEdge_CUDA->Location = System::Drawing::Point(640, 560);
            this->btEdge_CUDA->Size = System::Drawing::Size(150, 30);
            this->btEdge_CUDA->Text = L"Edge - CUDA";
            this->btEdge_CUDA->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btEdge_CUDA_Click);
            this->Controls->Add(this->btEdge_CUDA);

            this->btSharpen_CUDA = (gcnew System::Windows::Forms::Button());
            this->btSharpen_CUDA->Location = System::Drawing::Point(800, 560);
            this->btSharpen_CUDA->Size = System::Drawing::Size(150, 30);
            this->btSharpen_CUDA->Text = L"Sharpen - CUDA";
            this->btSharpen_CUDA->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btSharpen_CUDA_Click);
            this->Controls->Add(this->btSharpen_CUDA);

            this->Load += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::Pavic_gui_2024_Form_Load);
        }

        System::Void Pavic_gui_2024_Form_Load(System::Object^ sender, System::EventArgs^ e) {
            // No CUDA status UI
        }

        // Handlers
        System::Void btOpen_Click(System::Object^ sender, System::EventArgs^ e) {
            OpenFileDialog^ ofd = gcnew OpenFileDialog();
            ofd->Filter = "Image Files|*.bmp;*.png;*.jpg;*.jpeg";
            if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
                this->pbox_input->ImageLocation = ofd->FileName;
            }
        }

        System::Void btGaussian_CPU_Click(System::Object^ sender, System::EventArgs^ e) {
            if (this->pbox_input->Image == nullptr) { MessageBox::Show("Please open an image first.", "No Image", MessageBoxButtons::OK); return; }
            Bitmap^ inputImage = (Bitmap^)this->pbox_input->Image;
            auto t0 = high_resolution_clock::now();
            Bitmap^ out = nullptr;
            try { out = processor->ApplyGaussian_CPU(inputImage); }
            catch (Exception^ ex) { MessageBox::Show(ex->Message, "Error", MessageBoxButtons::OK); return; }
            auto t1 = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(t1 - t0);
            this->pbox_output->Image = out;
            this->textB_Time->Text = "Gaussian CPU: " + dur.count().ToString() + " ms";
        }

        System::Void btEdge_CPU_Click(System::Object^ sender, System::EventArgs^ e) {
            if (this->pbox_input->Image == nullptr) { MessageBox::Show("Please open an image first.", "No Image", MessageBoxButtons::OK); return; }
            Bitmap^ inputImage = (Bitmap^)this->pbox_input->Image;
            auto t0 = high_resolution_clock::now();
            Bitmap^ out = nullptr;
            try { out = processor->ApplyEdge_CPU(inputImage); }
            catch (Exception^ ex) { MessageBox::Show(ex->Message, "Error", MessageBoxButtons::OK); return; }
            auto t1 = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(t1 - t0);
            this->pbox_output->Image = out;
            this->textB_Time->Text = "Edge CPU: " + dur.count().ToString() + " ms";
        }

        System::Void btSharpen_CPU_Click(System::Object^ sender, System::EventArgs^ e) {
            if (this->pbox_input->Image == nullptr) { MessageBox::Show("Please open an image first.", "No Image", MessageBoxButtons::OK); return; }
            Bitmap^ inputImage = (Bitmap^)this->pbox_input->Image;
            auto t0 = high_resolution_clock::now();
            Bitmap^ out = nullptr;
            try { out = processor->ApplySharpen_CPU(inputImage); }
            catch (Exception^ ex) { MessageBox::Show(ex->Message, "Error", MessageBoxButtons::OK); return; }
            auto t1 = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(t1 - t0);
            this->pbox_output->Image = out;
            this->textB_Time->Text = "Sharpen CPU: " + dur.count().ToString() + " ms";
        }

        System::Void btGaussian_CUDA_Click(System::Object^ sender, System::EventArgs^ e) {
            if (this->pbox_input->Image == nullptr) { MessageBox::Show("Please open an image first.", "No Image", MessageBoxButtons::OK); return; }
            Bitmap^ inputImage = (Bitmap^)this->pbox_input->Image;
            auto t0 = high_resolution_clock::now();
            Bitmap^ out = nullptr;
            try { out = processor->ApplyGaussian_CUDA(inputImage); }
            catch (Exception^ ex) { MessageBox::Show(ex->Message, "CUDA Error", MessageBoxButtons::OK); return; }
            auto t1 = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(t1 - t0);
            this->pbox_output->Image = out;
            this->textB_Time->Text = "Gaussian CUDA: " + dur.count().ToString() + " ms";
        }

        System::Void btEdge_CUDA_Click(System::Object^ sender, System::EventArgs^ e) {
            if (this->pbox_input->Image == nullptr) { MessageBox::Show("Please open an image first.", "No Image", MessageBoxButtons::OK); return; }
            Bitmap^ inputImage = (Bitmap^)this->pbox_input->Image;
            auto t0 = high_resolution_clock::now();
            Bitmap^ out = nullptr;
            try { out = processor->ApplyEdge_CUDA(inputImage); }
            catch (Exception^ ex) { MessageBox::Show(ex->Message, "CUDA Error", MessageBoxButtons::OK); return; }
            auto t1 = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(t1 - t0);
            this->pbox_output->Image = out;
            this->textB_Time->Text = "Edge CUDA: " + dur.count().ToString() + " ms";
        }

        System::Void btSharpen_CUDA_Click(System::Object^ sender, System::EventArgs^ e) {
            if (this->pbox_input->Image == nullptr) { MessageBox::Show("Please open an image first.", "No Image", MessageBoxButtons::OK); return; }
            Bitmap^ inputImage = (Bitmap^)this->pbox_input->Image;
            auto t0 = high_resolution_clock::now();
            Bitmap^ out = nullptr;
            try { out = processor->ApplySharpen_CUDA(inputImage); }
            catch (Exception^ ex) { MessageBox::Show(ex->Message, "CUDA Error", MessageBoxButtons::OK); return; }
            auto t1 = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(t1 - t0);
            this->pbox_output->Image = out;
            this->textB_Time->Text = "Sharpen CUDA: " + dur.count().ToString() + " ms";
        }
    };

} // namespace pavicgui2024
