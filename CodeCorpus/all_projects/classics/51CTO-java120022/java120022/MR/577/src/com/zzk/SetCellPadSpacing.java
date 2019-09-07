package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetCellPadSpacing {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置单元格的填充和行间距.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfPTable table = new PdfPTable(2);// 定义表格
            table.addCell("no Padding");// 向单元格添加内容
            table.addCell("MingRiCompany MingRi MingRiCompany");// 单元格填充前内容
            table.addCell("Set Padding");// 向单元格添加内容
            table.getDefaultCell().setPadding(24);// 向单元格设置填充24
            table.addCell("MingRi MingRiCompany");// 单元格填充后内容
            document.add(table);// 将表格添加到文档
            PdfPTable table1 = new PdfPTable(2);// 定义表格
            table1.addCell("no Leading");// 向单元格添加内容
            table1
                    .addCell("MingRi MingRi MingRiCompanyMingRiCompany MingRiCompany");// 添加行间距前内容
            table1.getDefaultCell().setLeading(12, 1);// 添加行间距
            table1.addCell("Set Leading");// 向单元格添加内容
            table1
                    .addCell("MingRi MingRi MingRiCompanyMingRiCompany MingRiCompany");// 添加行间距后内容
            document.add(table1);// 将表格添加到文档
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
