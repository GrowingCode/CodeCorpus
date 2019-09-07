package com.zzk;

import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;
import java.io.IOException;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPageable;

public class PrintPDF {
    
    public static void main(String args[]) throws IOException,
            IllegalArgumentException, PrinterException {
        PrinterJob job = PrinterJob.getPrinterJob();// 创建打印作业
        PDDocument document = PDDocument.load("创建表格.pdf");// 获取待打印的文档
        Printable printable = new PDPageable(document);// 创建Printable对象
        job.setPrintable(printable);// 设置打印工作
        job.print();// 打印
        
    }
}
