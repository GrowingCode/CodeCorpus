package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class PdfInsertTable {
    public static void main(String[] args) {
        
        try {
            Document document = new Document();
            PdfWriter.getInstance(document, new FileOutputStream(
                    "c:\\Pdf嵌套表格.pdf"));
            document.open();// 打开文档
            PdfPTable table = new PdfPTable(4);// 创建表格对象
            PdfPTable table1 = new PdfPTable(2);// 创建表格对象
            table1.addCell("1.1");// 添加单元格内容
            table1.addCell("1.2");// 添加单元格内容
            PdfPTable table2 = new PdfPTable(1);// 创建表格对象
            table2.addCell("2.1");// 添加单元格内容
            table2.addCell("2.2");// 添加单元格内容
            table.addCell("table1");// 添加单元格内容
            table.addCell("table2");// 添加单元格内容
            table.addCell("text");// 添加单元格内容
            table.addCell("text");// 添加单元格内容
            table.addCell(table1);// 添加嵌套的表格
            table.addCell(table2);// 添加嵌套的表格
            table.addCell("cell");// 添加单元格内容
            table.addCell("cell");// 添加单元格内容
            document.add(table);// 将表格添加到文档中
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
    
}
