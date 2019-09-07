package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetCellAlignment {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置单元格的对齐方式.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfPTable table = new PdfPTable(2);// 定义表格
            Paragraph p = new Paragraph("I think Bale will win");// 定义段落和内容
            table.addCell("alignment left");// 向单元格添加内容
            PdfPCell cell = new PdfPCell(p);// 定义单元格
            cell.setHorizontalAlignment(Element.ALIGN_LEFT);// 设置单元格水平向左对齐
            table.addCell(cell);
            table.addCell("alignment right");// 向单元格添加内容
            PdfPCell cell1 = new PdfPCell(p);// 定义单元格
            cell1.setHorizontalAlignment(Element.ALIGN_RIGHT);// 设置单元格水平向右对齐
            table.addCell(cell1);
            table.addCell("alignment justified");// 向单元格添加内容
            PdfPCell cell2 = new PdfPCell(p);// 定义单元格
            cell2.setHorizontalAlignment(Element.ALIGN_JUSTIFIED);// 设置单元格为合理的对齐方式
            table.addCell(cell2);
            table.addCell("alignment center");// 向单元格添加内容
            PdfPCell cell3 = new PdfPCell(p);// 定义单元格
            cell3.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置单元格水平向中间对齐
            table.addCell(cell3);
            document.add(table);// 将表格添加到文档
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
