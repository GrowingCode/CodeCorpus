package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.FontFactory;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetCellHeight {
    
    public static void main(String[] args) {
        
        try {
            Font font = FontFactory.getFont("COURIER", 10, Font.BOLD);// 定义一个字体
            Font xfont = FontFactory.getFont("HELVETICA", 10, Font.BOLD);// 定义一个字体
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置单元格的高度.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfPTable table = new PdfPTable(2);// 定义表格
            table.getDefaultCell().setBackgroundColor(BaseColor.ORANGE);
            table.addCell(new Paragraph("default height", xfont));// 向表格添加单元格
            PdfPCell cell = new PdfPCell(new Paragraph("AAA", font));// 定义一个表格单元
            table.addCell(cell);// 向表格添加单元格
            table.addCell(new Paragraph("set height", xfont));
            PdfPCell cell2 = new PdfPCell(new Paragraph("ABC", font));// 定义单元格
            cell2.setFixedHeight(60);// 设置单元格高度为60
            table.addCell(cell2);// 将单元加入到表格
            table.addCell(new Paragraph("minimum height", xfont));
            PdfPCell cell3 = new PdfPCell(new Paragraph(
                    "A cat may look at a king.", font));// 定义单元格
            cell3.setMinimumHeight(40);// 设置单元格高度为40
            table.addCell(cell3);// 将单元加入到表格
            document.add(table);// 将表格添加到文档
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
